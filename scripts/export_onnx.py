"""SPICE Pre-train 推理模型 → ONNX 导出 + 数值对拍（Keras vs onnxruntime）。

产出: export/spice_infer_fixed.onnx（见 docs/ONNX_USAGE.md）
用法: PYTHONPATH=. python scripts/export_onnx.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spice_pre.config import load_config
from spice_pre.models import SPICEPretrainModel
from spice_pre.data.preprocessing import seq_to_tokens, normalize_env

CFG = os.path.join(ROOT, "configs/pretrain.yaml")
W = os.path.join(ROOT, "checkpoints/pretrain/best_weights.weights.h5")
OUT = os.path.join(ROOT, "export")
AA = "ACDEFGHIKLMNPQRSTVWY"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config(CFG)
    # 推理导出用 fp32 einsum：与训练 fp16 同权重、数学等价，ONNX 更干净可移植（GUI 不需要 fp16）
    cfg.model.distogram_fp16 = False
    tf.keras.mixed_precision.set_global_policy("float32")

    model = None
    # 全头导出：Head A(坐标) + B(突变) + Bp(突变坐标=Head A折叠, 别名) + C(环境偏移) + D(置信度) + distogram
    # 注意：B/C/D 在预训练 checkpoint 中无权重（RL 阶段才训）→ 随机初始化，skip_mismatch 跳过；
    # Bp 无独立权重，coords_mut 即 Head A 对(突变)序列的折叠。
    # 契约先锁死供 GUI 开发；RL 权重出来后重跑本脚本即得可用全头 ONNX（契约不变）。
    HEADS = ("A", "B", "Bp", "C", "D")
    model = SPICEPretrainModel(cfg.model, heads=HEADS)
    model({"tokens": tf.zeros([1, 8], tf.int32), "env": tf.zeros([1, 3]),
           "mask": tf.ones([1, 8])}, training=False)
    model.load_weights(W, skip_mismatch=True)  # ⚠️ skip_mismatch：B/Bp/C/D 不存在于预训练 checkpoint
    print(f"模型加载 OK | params={sum(v.shape.num_elements() for v in model.trainable_variables):,} "
          f"| heads={HEADS}")

    @tf.function(input_signature=[
        tf.TensorSpec([None, None], tf.int32, name="tokens"),
        tf.TensorSpec([None, 3], tf.float32, name="env"),
        tf.TensorSpec([None, None], tf.float32, name="mask"),
    ])
    def infer(tokens, env, mask):
        out = model({"tokens": tokens, "env": env, "mask": mask}, training=False)
        return {
            "coords": out["coords"],
            "dist_logits": out["dist_logits"],
            "mutation": out["mutation"],
            "coords_mut": out["coords_mut"],
            "env_offset": out["env_offset"],
            "conf": out["conf"],
        }

    sm_path = os.path.join(OUT, "saved")
    tf.saved_model.save(model, sm_path, signatures={"serving_default": infer})
    print("SavedModel 保存 OK")

    import tf2onnx
    raw_path = os.path.join(OUT, "spice_infer_raw.onnx")
    spec = [
        tf.TensorSpec([None, None], tf.int32, name="tokens"),
        tf.TensorSpec([None, 3], tf.float32, name="env"),
        tf.TensorSpec([None, None], tf.float32, name="mask"),
    ]
    tf2onnx.convert.from_function(infer, input_signature=spec,
                                  output_path=raw_path, opset=13)
    print("tf2onnx 转换 OK")

    # ONNX 图后处理：修复 tf2onnx 不支持的 Erfc / Cross 算子
    import onnx
    from onnx import helper, numpy_helper
    fixed_path = os.path.join(OUT, "spice_infer_fixed.onnx")
    m = onnx.load(raw_path)
    g = m.graph
    new_nodes, new_init = [], []
    _n = [0]

    def _idx(prefix, i):
        _n[0] += 1
        name = f"{prefix}/idx{_n[0]}"
        new_init.append(numpy_helper.from_array(np.array([i], np.int64), name=name))
        return name

    for node in g.node:
        if node.op_type == "Erfc":
            x, out = node.input[0], node.output[0]
            erf = helper.make_node("Erf", [x], [out + "_erf"], name=node.name + "/erf")
            _n[0] += 1
            one_n = f"{node.name}/one{_n[0]}"
            new_init.append(numpy_helper.from_array(np.array(1.0, np.float32), name=one_n))
            sub = helper.make_node("Sub", [one_n, out + "_erf"], [out], name=node.name + "/sub")
            new_nodes += [erf, sub]
        elif node.op_type == "Cross":
            u, v, out = node.input[0], node.input[1], node.output[0]

            def _g(t, i):
                _n[0] += 1
                o = f"{out}/g{_n[0]}"
                new_nodes.append(helper.make_node("Gather", [t, _idx(node.name, i)], [o], axis=-1))
                return o

            u0, u1, u2 = _g(u, 0), _g(u, 1), _g(u, 2)
            v0, v1, v2 = _g(v, 0), _g(v, 1), _g(v, 2)

            def _mulsub(a, b, c, d, tag):
                m1, s = f"{out}/{tag}m", f"{out}/{tag}s"
                new_nodes.append(helper.make_node("Mul", [a, b], [m1]))
                new_nodes.append(helper.make_node("Mul", [c, d], [s]))
                o = f"{out}/{tag}"
                new_nodes.append(helper.make_node("Sub", [m1, s], [o]))
                return o

            x = _mulsub(u1, v2, u2, v1, "x")
            y = _mulsub(u2, v0, u0, v2, "y")
            z = _mulsub(u0, v1, u1, v0, "z")
            new_nodes.append(helper.make_node("Concat", [x, y, z], [out], axis=-1))
        else:
            new_nodes.append(node)
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(new_init)
    onnx.checker.check_model(m)
    onnx.save(m, fixed_path)
    os.remove(raw_path)
    print(f"ONNX 图修复 OK（Erfc→1-Erf，Cross→分解）-> {fixed_path}")

    # 数值对拍：Keras vs onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(fixed_path, providers=["CPUExecutionProvider"])
    print("ONNX inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("ONNX outputs:", [(o.name, o.shape) for o in sess.get_outputs()])

    def _rand_protein(L):
        seq = "".join(np.random.choice(list(AA), L))
        tokens = seq_to_tokens(seq).astype(np.int32)[None]
        env = normalize_env(None, None, None)[None].astype(np.float32)
        mask = np.ones((1, L), np.float32)
        return tokens, env, mask

    print("\n=== 数值对拍（Keras vs ONNX，max abs diff）===")
    ok = True
    for L in (60, 150, 300):
        tokens, env, mask = _rand_protein(L)
        ko = model({"tokens": tokens, "env": env, "mask": mask}, training=False)
        kn = {k: ko[k].numpy() for k in (
            "coords", "dist_logits", "mutation", "coords_mut", "env_offset", "conf")}
        on = sess.run(None, {"tokens": tokens, "env": env, "mask": mask})
        od = {o.name: a for o, a in zip(sess.get_outputs(), on)}
        row_ok = True
        for k, kv in kn.items():
            d = np.max(np.abs(od[k] - kv))
            tol = 1e-2 if k == "dist_logits" else 1e-3
            good = bool(np.isfinite(d) and d < tol)
            row_ok &= good
            print(f"    {k:>12}: maxdiff={d:.2e} {'✅' if good else '❌'}")
        ok &= row_ok
        print(f"  L={L:4d}: {'✅' if row_ok else '❌'}")

    print("\n" + ("✅ 对拍通过：ONNX 与 Keras 输出一致（含 B/Bp/C/D，契约已锁死）" if ok
                  else "❌ 对拍失败：需排查导出"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
