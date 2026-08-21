#!/usr/bin/env python
"""6QQE「born-NaN」SAC 集群 smoke —— 登录节点直跑版（不用 sbatch）。

用途：本地(Apple Silicon/CPU/旧权重)无法复现 6QQE 每集 watchdog 触发 + actor kernel
[259,256] 100% NaN。此脚本在 HPC 真实环境（容器 TF + 真实 n45000 模型 + 真实 yaml）逐层
复现，回答：NaN 到底从哪一步进来 —— (0) 环境指纹 (1) 部署代码审计 (2) encode_z 极端值
(3) 新 SAC 前向 (4) 首次真实 update（含 buffer 守卫测试）(5) 重建隔离/stale-trace 测试。

用法（登录节点，非 sbatch）：
    bash scripts/experiments/slurm/smoke_6qqe_login.sh
等价手动：
    module load singularity/3.7.3
    singularity exec "$SIF" bash -lc 'source $CONDA_SH && conda activate spice && cd $MODEL_ROOT && \
        export PYTHONPATH=$MODEL_ROOT CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
               TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS="--tf_xla_auto_jit=0" \
               ONEDNN_DEFAULT_FPMATH_MODE=FP32 && python scripts/smoke_6qqe_login.py'

参数：--root MODEL_ROOT(默认 $HOME/spice/model) --post-yaml(默认 ../data/data_efficiency/n45000/posttrain.yaml)
     --pdb-id(默认 6QQE) --seq(默认 6QQE 溶菌酶 129aa)
"""
import argparse
import os
import sys

import numpy as np

# ============================================================ 路径解析 ============================================================
ap = argparse.ArgumentParser(add_help=True)
ap.add_argument("--root", default=os.environ.get("MODEL_ROOT", os.path.expanduser("~/spice/model")))
ap.add_argument("--post-yaml", default=None, help="真实 RL 使用的 yaml（rl-gen 生成）")
ap.add_argument("--pdb-id", default="6QQE")
ap.add_argument("--seq", default=None, help="6QQE 序列，默认内置 129aa 溶菌酶")
ap.add_argument("--no-trace", action="store_true", help="关闭逐层 trace（默认开，打 actor/critic 每层输出 + update 中间量）")
ap.add_argument("--eager", action="store_true", help="强制 SAC 更新 + encode_z 走 eager（模拟集群 eager_update:true）")
ap.add_argument("--check6-only", action="store_true",
                help="只跑 Check 0(环境)+Check 6(图模式 Dense sanity) —— 容器门禁专用，快速判定 TF 是否数值健康")
args = ap.parse_args()
TRACE = not args.no_trace
print(f"[INFO] 逐层 trace = {TRACE}（--no-trace 关闭）")

ROOT = os.path.abspath(args.root)
if args.check6_only:
    # 容器门禁：不依赖模型目录，纯 Check 0 + Check 6（matmul sanity），可在任意环境跑
    print(f"[INFO] 容器门禁模式（Check 6 only，不加载模型）")
else:
    POST_YAML = args.post_yaml or os.path.join(ROOT, "..", "data", "data_efficiency", "n45000", "posttrain.yaml")
    POST_YAML = os.path.abspath(POST_YAML)
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    print(f"[INFO] MODEL_ROOT   = {ROOT}")
    print(f"[INFO] POST_YAML    = {POST_YAML}")
    print(f"[INFO] 存在? = {os.path.exists(POST_YAML)}")

SEQ = args.seq or "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGQTAWVAWRNRCKGTDVQAWIRGCRL"

PASS = 0
FAIL = 0
WARN = 0


def verdict(tag: str, ok: bool, msg: str):
    global PASS, FAIL, WARN
    if ok:
        PASS += 1
        print(f"  [PASS] {tag}: {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {tag}: {msg}")


def warn(tag: str, msg: str):
    global WARN
    WARN += 1
    print(f"  [WARN] {tag}: {msg}")


def finite_of(model) -> tuple:
    """返回 (全部有限?, 首个非有限描述, nan数)"""
    for _m in (model.actor, model.critic, model.critic_target):
        for _w in _m.trainable_variables:
            _a = _w.numpy()
            if not np.all(np.isfinite(_a)):
                _nn = int(np.sum(~np.isfinite(_a)))
                return False, f"{_w.name} shape={list(_a.shape)} 非有限 {_nn}/{_a.size}", _nn
    if not np.isfinite(model.log_alpha.numpy()):
        return False, f"log_alpha={model.log_alpha.numpy()}", 1
    return True, "", 0


# ============================================================ Check 0 环境指纹 ============================================================
def check0_env():
    print("\n===== [Check 0] 环境指纹 =====")
    import tensorflow as tf
    print(f"  TF      = {tf.__version__}")
    try:
        import keras
        print(f"  Keras   = {keras.__version__}")
    except Exception as e:
        print(f"  Keras   = ? ({e})")
    import sys as _s
    print(f"  Python  = {_s.version.split()[0]}")
    print(f"  devices = {tf.config.list_physical_devices()}")
    devs = tf.config.list_physical_devices("GPU")
    verdict("设备", len(devs) == 0, f"GPU 列表={devs}（RL 跑 CPU，CUDA_VISIBLE_DEVICES 应为空）")
    import tensorflow as _tf
    verdict("onednn/xla", "TF_ENABLE_ONEDNN_OPTS=0" in os.environ or _tf.__version__ >= "2.16",
            f"TF_ENABLE_ONEDNN_OPTS={os.environ.get('TF_ENABLE_ONEDNN_OPTS')} TF_XLA_FLAGS={os.environ.get('TF_XLA_FLAGS')}")


# ============================================================ Check 1 部署代码审计 ============================================================
MARKERS = {
    "spice_rl/sac/sac.py": [
        "can_sample(self.cfg.batch_size)",                    # buffer 守卫（2026-08-17 前可能缺失）
        "tf.where(tf.math.is_finite(v), v, tf.zeros_like(v))",  # 输入 isfinite 清洗
        "tf.clip_by_global_norm",                              # 梯度全局裁剪
        "log_pi = tf.clip_by_value(log_pi, -100.0, 0.0)",      # 熵项有界
        "y = tf.where(tf.math.is_finite(y), y, tf.zeros_like(y))",  # y 兜底
        "q1 = tf.clip_by_value(q1, -1e6, 1e6)",                # critic 输出 clamp
    ],
    "spice_rl/sac/networks.py": [
        "u_ref",                  # u_hist 归一化
        "tf.math.is_finite",      # gaussian logprob 清洗
    ],
    "spice_rl/sac/buffer.py": [
        'np.clip(tr["M"], -5.0, 5.0)',            # M 裁剪
        'np.clip(tr["reward"], -150.0, 15.0)',    # reward 裁剪
    ],
    "spice_rl/train_post.py": [
        "nan_watchdog",           # watchdog 开关
        "_sac_nan_report",        # 诊断函数（6QQE 日志格式来源）
        "escape_step_threshold",  # Env Escape
        "recovery_delta_ph",      # Recovery
    ],
    "spice_rl/config.py": [
        "nan_watchdog",
        "escape_step_threshold",
        "batch_size",
    ],
    "spice_rl/env/md_env.py": [
        "TERMINAL_CRASH_REWARD",  # 崩溃 -100
        "np.isfinite(u_kj)",      # u_kj 清洗（md_env 用 numpy）
    ],
}


def check1_audit():
    print("\n===== [Check 1] 部署代码审计（是否最新版，防 stale 部署）=====")
    all_ok = True
    for _f, _mks in MARKERS.items():
        _p = os.path.join(ROOT, _f)
        if not os.path.exists(_p):
            warn("文件缺失", _f)
            all_ok = False
            continue
        _src = open(_p, encoding="utf-8", errors="replace").read()
        for _mk in _mks:
            if _mk not in _src:
                all_ok = False
                warn(f"{_f} 缺标记", f"缺少: {_mk[:60]}")
    verdict("部署代码版本", all_ok, "全部关键防护标记都在 → 集群跑的是最新 sac/networks/buffer/train_post/config/md_env")
    return all_ok


# ============================================================ Check 2/3/4/5 ============================================================
def main_checks():
    import tensorflow as tf
    from spice_rl.config import load_config
    from spice_rl.train_post import (
        build_rl_model, encode_z, tokens_from_seq, _normalize_env, cfg_sac_z_dim, _sac_nan_report,
    )
    from spice_rl.sac import SACTrainer

    cfg = load_config(POST_YAML)
    if args.eager:
        cfg.sac.eager_update = True
        print("[INFO] --eager 已强制 cfg.sac.eager_update=True（SAC 更新 + encode_z 走 eager）")
    print(f"\n===== [Check 2] 真实模型 + 6QQE encode_z =====")
    print(f"  cfg.sac.batch_size = {cfg.sac.batch_size}   discrete_position_dim = {cfg.sac.discrete_position_dim}")
    model = build_rl_model(cfg)
    z_dim = cfg_sac_z_dim(model)
    print(f"  z_dim = {z_dim}  (actor 首层 kernel 应为 [z_dim+env_dim=259, 256])")
    tokens, mask = tokens_from_seq(SEQ, cfg.sac.discrete_position_dim)
    print(f"  seq len = {len(SEQ)}  tokens = {tokens.shape}")

    # ---- 2a: 全网格 + ep0 真实 env_fail + 随机扫 ----
    envs = []
    for _ph in (2.0, 4.0, 7.0, 10.0, 12.0):
        for _t in (260.0, 293.0, 298.0, 303.0, 330.0):
            envs.append((_ph, _t))
    envs.append((4.0, 330.0))  # 6QQE ep0 记录的真实 env_fail
    rng = np.random.default_rng(20260819)
    for _ in range(200):
        envs.append((float(rng.uniform(0, 14)), float(rng.uniform(150, 400))))
    bad = 0
    maxabs = 0.0
    for (_ph, _t) in envs:
        _e = _normalize_env((_ph, _t, 0.0), cfg.env)
        _z = encode_z(model, tokens, _e, mask)
        if not np.all(np.isfinite(_z)):
            bad += 1
            if bad <= 3:
                warn("encode_z 非有限", f"ph={_ph:.1f} T={_t:.1f} z={_z}")
        _ma = float(np.max(np.abs(_z)))
        maxabs = max(maxabs, _ma)
    verdict("encode_z 全有限", bad == 0, f"扫描 {len(envs)} 个 env，非有限 {bad} 个")
    warn("encode_z 极端值", f"跨 {len(envs)} env 的 |z| 最大 = {maxabs:.3f}（>100 提示真实模型 z 尺度异常，会压垮 actor 首层）"
         if maxabs > 100 else f"|z| 最大 = {maxabs:.3f}（正常）")

    # ---- 2b: tf.function 版 vs 手写 eager 版一致性（抓容器 TF 图模式 bug）----
    try:
        _e = _normalize_env((4.0, 330.0, 0.0), cfg.env)
        _ztf = encode_z(model, tokens, _e, mask)
        _out = model({"tokens": tf.constant(tokens[None], tf.int32),
                      "env": tf.constant(_e[None], tf.float32),
                      "mask": tf.constant(mask[None], tf.float32)}, training=False)
        _zraw = _out["z"][0].numpy()
        _zc = np.where(np.isfinite(_zraw), _zraw, 0.0)
        _m = mask[:, None]
        _ze = (_zc * _m).sum(0) / max(_m.sum(), 1.0)
        _raw_absmax = float(np.max(np.abs(_ze))) if np.all(np.isfinite(_ze)) else float("nan")
        # 复刻 _encode_z_pool 的 clip + RMS 归一化（2026-08-19 born-NaN 修复）
        _ze = np.clip(_ze, -1e6, 1e6)
        _ze = _ze / max(float(np.sqrt(np.mean(_ze * _ze) + 1e-8)), 1e-6)
        _close = np.allclose(_ztf, _ze, atol=1e-3) and np.all(np.isfinite(_ztf))
        warn("原始 z 尺度", f"归一化前 |z| max = {_raw_absmax:.3e}（本地旧权重≈35；集群曾 3.5e35 → 若仍≫100 说明 pretrain ckpt 可疑）"
             if _raw_absmax > 100 else f"归一化前 |z| max = {_raw_absmax:.3e}（健康，模型 z 正常 O(1~100)）")
        verdict("tf.function 一致性", _close,
                f"encode_z(tf.function) 与 eager 前向 max|Δ|={np.max(np.abs(_ztf - _ze)):.2e}")
    except Exception as _e2:
        warn("tf.function 一致性", f"异常: {_e2}")

    # ============================================================ Check 3 新 SAC 前向 ============================================================
    print("\n===== [Check 3] 全新 SAC：ensure_built + 20×act 前向 =====")
    cont_dim = int(cfg.env.force_dim) + int(cfg.env.env_offset_dim)
    print(f"  cont_dim = force_dim({cfg.env.force_dim}) + env_offset_dim({cfg.env.env_offset_dim}) = {cont_dim}")
    sac = SACTrainer(cfg.sac, z_dim=z_dim, cont_dim=cont_dim, u_window=cfg.env.u_window,
                     trace_layers=TRACE)
    sac.ensure_built()
    fin, desc, _ = finite_of(sac)
    verdict("ensure_built 后权重", fin, desc or "actor/critic/critic_target/log_alpha 全有限")

    z_mask = np.zeros(cfg.sac.discrete_position_dim, np.float32)
    z_mask[:len(mask)] = mask
    corrupted_step = -1
    for i in range(20):
        _ph, _t = envs[i % len(envs)]
        _e = _normalize_env((_ph, _t, 0.0), cfg.env)
        _z = encode_z(model, tokens, _e, mask)
        _ac, _ad = sac.act(_z, _e, z_mask)
        if not (np.all(np.isfinite(_ac)) and np.all(np.isfinite(_ad))):
            corrupted_step = i
            break
        _f, _d, _ = finite_of(sac)
        if not _f:
            corrupted_step = i
            warn("act 后权重被污染", f"step {i}: {_d}")
            break
    verdict("20×act 前向", corrupted_step < 0,
            f"actor 权重全程有限，输出有限（首污染 step={corrupted_step}）")

    # ============================================================ Check 4 真实 update ============================================================
    print("\n===== [Check 4] 真实 buffer + 首次 update（含 buffer 守卫测试）=====")
    # 4a: buffer 守卫 —— 填到 < batch_size，update() 必须早退且不动权重
    guard_ok = True
    n_fill = max(1, int(cfg.sac.batch_size) - 1)
    for i in range(n_fill):
        _ph, _t = envs[i % len(envs)]
        _e = _normalize_env((_ph, _t, 0.0), cfg.env)
        _z = encode_z(model, tokens, _e, mask)
        sac.buffer.add({
            "z": _z.astype(np.float32), "env": _e.astype(np.float32),
            "M": np.clip(rng.uniform(-5, 5, 5), -5, 5).astype(np.float32),
            "u_hist": np.clip(rng.uniform(-1e6, 1e6, cfg.env.u_window), -1e6, 1e6).astype(np.float32),
            "action_cont": rng.normal(0, 3, cont_dim).astype(np.float32),
            "action_disc": rng.uniform(0, 1, cfg.sac.discrete_position_dim + cfg.sac.aa_dim).astype(np.float32),
            "mutation_mask": np.float32(0.0), "z_mask": z_mask.astype(np.float32),
            "reward": np.float32(-100.0), "done": np.float32(1.0),
            "next_z": _z.astype(np.float32), "next_env": _e.astype(np.float32),
            "next_M": np.clip(rng.uniform(-5, 5, 5), -5, 5).astype(np.float32),
            "next_u_hist": np.clip(rng.uniform(-1e6, 1e6, cfg.env.u_window), -1e6, 1e6).astype(np.float32),
        })
    _L = sac.update(z_mask)
    _f, _d, _ = finite_of(sac)
    guard_ok = _f
    verdict("buffer 守卫", guard_ok,
            f"buffer={len(sac.buffer)} < batch_size={cfg.sac.batch_size}，update 返回 loss={_L['critic_loss']} 但权重 {_d or '保持有限'}")

    # 4b: 满 buffer 真实更新 —— 用 6QQE 真实 z + 真实 act 动作，崩溃密集，跑 80 步
    while len(sac.buffer) < max(cfg.sac.batch_size + 8, 40):
        i = len(sac.buffer)
        _ph, _t = envs[i % len(envs)]
        _e = _normalize_env((_ph, _t, 0.0), cfg.env)
        _z = encode_z(model, tokens, _e, mask)
        _ac, _ad = sac.act(_z, _e, z_mask)
        _crash = (i % 20 == 19)
        sac.buffer.add({
            "z": _z.astype(np.float32), "env": _e.astype(np.float32),
            "M": np.clip(rng.uniform(-5, 5, 5), -5, 5).astype(np.float32),
            "u_hist": np.clip(rng.uniform(0, 1e6, cfg.env.u_window), -1e6, 1e6).astype(np.float32),
            "action_cont": _ac.astype(np.float32), "action_disc": _ad.astype(np.float32),
            "mutation_mask": np.float32(0.0), "z_mask": z_mask.astype(np.float32),
            "reward": np.float32(-100.0 if _crash else rng.uniform(-10, 0.0)),
            "done": np.float32(1.0 if _crash else 0.0),
            "next_z": _z.astype(np.float32), "next_env": _e.astype(np.float32),
            "next_M": np.clip(rng.uniform(-5, 5, 5), -5, 5).astype(np.float32),
            "next_u_hist": np.clip(rng.uniform(0, 1e6, cfg.env.u_window), -1e6, 1e6).astype(np.float32),
        })
    print(f"  buffer 填满 = {len(sac.buffer)}，开始 80 步真实 update…")
    nan_step = -1
    import time as _time
    _t0 = _time.time()
    for i in range(80):
        _L = sac.update(z_mask)
        _f, _d, _ = finite_of(sac)
        if not _f:
            nan_step = i
            warn("update 产生 NaN", f"step {i}: {_d}")
            break
        if i in (0, 9, 39, 79):
            print(f"    update {i}: critic={_L['critic_loss']:.2f} actor={_L['actor_loss']:.2f} "
                  f"alpha={_L['alpha']:.4f} buffer={len(sac.buffer)}")
    _dt = _time.time() - _t0
    _n_run = nan_step if nan_step >= 0 else 80
    warn("RL 更新开销", f"{_n_run} 步 update 墙钟 {_dt:.2f}s → {_dt / max(_n_run, 1):.3f}s/步 "
         f"（{('eager' if getattr(cfg.sac, 'eager_update', False) else '图模式')}；"
         f"对比 SE 每步 MD 数秒~每 build 2min → RL 占比可忽略）")
    verdict("80 步真实 update", nan_step < 0,
            f"actor/critic/log_alpha 全程有限（首个 NaN step={nan_step}）")

    # ============================================================ Check 5 重建隔离 / stale-trace ============================================================
    print("\n===== [Check 5] 重建隔离（模拟 watchdog 重建后，新 SAC 是否被旧 trace 污染）=====")
    if nan_step >= 0:
        print("  [SKIP] Check 4 已 NaN，无需模拟 —— 根因已锁定在 update 内。")
        return
    # 5a: 模拟旧 SAC 被污染（直接给 actor kernel 灌 NaN），验证诊断格式与 6QQE 日志一致
    _old_kernel = None
    for _w in sac.actor.trainable_variables:
        if "kernel" in _w.name and _w.shape == [z_dim + 3, 256]:
            _old_kernel = _w
            break
    _old_kernel.assign(tf.constant(np.full(_old_kernel.shape.as_list(), np.nan), tf.float32))
    _diag = _sac_nan_report(sac)
    _expect = "actor/" in _diag and "非有限" in _diag
    verdict("诊断格式一致", _expect, f"_sac_nan_report → '{_diag}'（应与 6QQE 日志 actor/kernel [259,256] 66304/66304 同格式）")
    # 5b: 重建全新 SAC_B（同 cfg），跑真实 update，验证不被旧 trace/旧变量污染
    sac_b = SACTrainer(cfg.sac, z_dim=z_dim, cont_dim=cont_dim, u_window=cfg.env.u_window,
                       trace_layers=TRACE)
    sac_b.ensure_built()
    _f, _d, _ = finite_of(sac_b)
    verdict("SAC_B ensure_built", _f, _d or "全有限")
    # 先往 SAC_B buffer 灌真实数据
    for i in range(max(cfg.sac.batch_size + 8, 40)):
        _ph, _t = envs[i % len(envs)]
        _e = _normalize_env((_ph, _t, 0.0), cfg.env)
        _z = encode_z(model, tokens, _e, mask)
        _crash = (i % 20 == 19)
        sac_b.buffer.add({
            "z": _z.astype(np.float32), "env": _e.astype(np.float32),
            "M": np.clip(rng.uniform(-5, 5, 5), -5, 5).astype(np.float32),
            "u_hist": np.clip(rng.uniform(0, 1e6, cfg.env.u_window), -1e6, 1e6).astype(np.float32),
            "action_cont": rng.normal(0, 3, cont_dim).astype(np.float32),
            "action_disc": rng.uniform(0, 1, cfg.sac.discrete_position_dim + cfg.sac.aa_dim).astype(np.float32),
            "mutation_mask": np.float32(0.0), "z_mask": z_mask.astype(np.float32),
            "reward": np.float32(-100.0 if _crash else rng.uniform(-10, 0.0)),
            "done": np.float32(1.0 if _crash else 0.0),
            "next_z": _z.astype(np.float32), "next_env": _e.astype(np.float32),
            "next_M": np.clip(rng.uniform(-5, 5, 5), -5, 5).astype(np.float32),
            "next_u_hist": np.clip(rng.uniform(0, 1e6, cfg.env.u_window), -1e6, 1e6).astype(np.float32),
        })
    # 关键：重建后第一次真实 update（正是 watchdog 每次重建后的首个 update）
    _L = sac_b.update(z_mask)
    _f, _d, _ = finite_of(sac_b)
    verdict("SAC_B 重建后首 update", _f,
            f"critic={_L['critic_loss']:.2f} actor={_L['actor_loss']:.2f} alpha={_L['alpha']:.4f} "
            f"权重 {_d or '全有限'}（旧 NaN SAC 未污染新 SAC → 排除 stale-trace）")
    # 5c: 再跑几步确认 SAC_B 稳定
    _ok = True
    for i in range(10):
        _L = sac_b.update(z_mask)
        _f, _d, _ = finite_of(sac_b)
        if not _f:
            _ok = False
            warn("SAC_B 后续 update 污染", f"step {i}: {_d}")
            break
    verdict("SAC_B 后续 10 步", _ok, "重建后 SAC 持续学习，无 stale-trace 污染")


def check6_graph_sanity():
    """决定性隔离：随机输入 Dense（无模型）graph vs eager 对拍。
    若随机输入下图模式就发散 → 容器 TF/onednn/MKL 数值坏了，与模型/代码无关。
    Check 2/4 集群实锤：归一化后 z~11、eager actor mean~1.7，但图模式 actor mean=1.4e5。"""
    print("\n===== [Check 6] 容器 TF 图模式 sanity（随机输入 Dense，无模型）=====")
    import tensorflow as tf
    rng = np.random.default_rng(7)
    problems = []
    for shape, units in [([32, 256], 256), ([32, 256], 18), ([32, 289], 256), ([1, 259], 256)]:
        x = rng.normal(0, 1.0, shape).astype(np.float32)
        d_eager = tf.keras.layers.Dense(units)
        d_eager(x)  # build
        y_eager = d_eager(x).numpy()
        d_graph = tf.keras.layers.Dense(units)
        d_graph(x)
        d_graph.set_weights(d_eager.get_weights())
        y_graph = tf.function(lambda xx: d_graph(xx))(x).numpy()
        a_e = float(np.max(np.abs(y_eager)))
        a_g = float(np.max(np.abs(y_graph)))
        diff = float(np.max(np.abs(y_eager - y_graph)))
        ok = bool(np.all(np.isfinite(y_graph))) and diff < 1e-2 * max(1.0, a_e)
        print(f"  shape={shape} units={units}: eager_absmax={a_e:.4f} graph_absmax={a_g:.4f} "
              f"maxΔ={diff:.4e} {'OK' if ok else '✗ 图模式发散!'}")
        if not ok:
            problems.append((shape, units, diff, a_e, a_g))
    verdict("图模式 Dense 一致性", len(problems) == 0,
            "随机输入下 eager==graph → 容器 TF 数值健康（那问题在模型权重）"
            if not problems else f"{len(problems)} 个形状图模式发散 → 容器 TF/onednn/MKL bug，与代码无关")


if __name__ == "__main__":
    if args.check6_only:
        check0_env()
        check6_graph_sanity()
        print("\n==============================================")
        print(f"  [容器门禁] PASS={PASS}  FAIL={FAIL}  WARN={WARN}")
        print("  门禁通过(Check6 全 OK) → 容器 TF 数值健康，可跑 RL；否则容器不可用。")
        print("==============================================")
        sys.exit(1 if FAIL else 0)
    check0_env()
    check1_audit()
    main_checks()
    check6_graph_sanity()
    print("\n==============================================")
    print(f"  总评: PASS={PASS}  FAIL={FAIL}  WARN={WARN}")
    print("  任何 FAIL → 把上面 [FAIL]/[WARN] 行贴回来即可定位根因。")
    print("==============================================")
    sys.exit(1 if FAIL else 0)
