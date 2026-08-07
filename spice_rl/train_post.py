"""双路后训练（Post-train / RL）主循环 —— Phase 2。

四环节流水线：
1. 预训练（已完成，加载权重）
2. 快筛：MD 短跑检查物理合法性（可跳过）
3. 双线程扰动 + 双路探索：
   - 路径 A（不突变）：SAC 微观环固定序列，一正一反扰动环境，探稳定区间/崩溃边界
   - 路径 B（可能突变）：ES 宏观环在 Head-B 采样 1~3 点突变，SAC 评估存活
4. 双路回流：
   - 路径 A 崩溃 → 记录 Env_fail，触发路径 B
   - 路径 B 存活 → 时间平均坐标作伪标签回流

用法：
    python -m spice_rl.train_post --config configs/posttrain.yaml [--max-episodes N]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import tensorflow as tf

from spice_rl.config import Config, load_config
from spice_rl.env import MDSimulationEnv
from spice_rl.env import (
    quick_check_env,
    save_phase_map,
    scan_phase_map,
    summarize_phase_map,
)
from spice_rl.confidence import ConfidenceHeadTrainer
from spice_rl.pseudo_labels import write_pseudo_tfrecord
from spice_rl.sac import SACTrainer

# 氨基酸字母（与 spice_pre.data.preprocessing 一致）
AA20 = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# 模型：加载完整双路模型（Head A/B/B'/C/D）+ Pre-train 权重
# ---------------------------------------------------------------------------
def build_rl_model(cfg: Config, max_seq_len: int = 512):
    from spice_pre.config import load_config as pre_load
    from spice_pre.models import SPICEPretrainModel

    # 复用 Pre-train 模型结构，启用全部头
    pre_cfg = pre_load(cfg.post.pretrain_config or "configs/pretrain.yaml")
    model = SPICEPretrainModel(
        pre_cfg.model, heads=("A", "B", "Bp", "C", "D")
    )
    # Keras 3 惰性构建
    model(
        {
            "tokens": tf.zeros([1, 8], tf.int32),
            "env": tf.zeros([1, 3]),
            "mask": tf.ones([1, 8]),
        },
        training=False,
    )
    ckpt = cfg.post.pretrain_ckpt
    if os.path.exists(ckpt):
        # skip_mismatch：只恢复 backbone + Head A（Head B/C/D 保持随机，交给 ES/SAC）
        model.load_weights(ckpt, skip_mismatch=True)
        print(f"已加载 Pre-train 权重: {ckpt}")
    return model


def encode_z(model, tokens, env, mask) -> np.ndarray:
    """序列+环境 → mean-pool 嵌入 z [D]。"""
    out = model(
        {
            "tokens": tf.constant(tokens.astype(np.int32)[None]),
            "env": tf.constant(np.asarray(env, np.float32)[None]),
            "mask": tf.constant(np.asarray(mask, np.float32)[None]),
        },
        training=False,
    )
    z = out["z"][0]                      # [L, D]
    m = tf.constant(np.asarray(mask, np.float32)[None])
    z_pool = tf.reduce_sum(z * m, axis=0) / tf.maximum(tf.reduce_sum(m), 1.0)
    return z_pool.numpy()


def tokens_from_seq(seq: str, max_len: int) -> tuple:
    """序列 → (tokens, mask, z_mask)。z_mask 补零到 discrete_position_dim。"""
    from spice_pre.data.preprocessing import seq_to_tokens

    tokens = seq_to_tokens(seq[:max_len])
    L = tokens.shape[0]
    mask = np.ones(L, np.float32)
    return tokens, mask


# ---------------------------------------------------------------------------
# 路径 B：突变后结构（模型 Head B' 输出坐标 → 引擎重建侧链）
# ---------------------------------------------------------------------------
_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}
_BACKBONE = {"N", "CA", "C", "O"}


def predict_mutant_coords(model, tokens, env, mask) -> np.ndarray:
    """Head B' 输出突变后 Cα 坐标 [L,3]（模型输出的突变后结构）。

    Pre-train 未训练 Head B' 时回退到 Head A。
    """
    out = model(
        {
            "tokens": tf.constant(tokens.astype(np.int32)[None]),
            "env": tf.constant(np.asarray(env, np.float32)[None]),
            "mask": tf.constant(np.asarray(mask, np.float32)[None]),
        },
        training=False,
    )
    if "coords_mut" in out:
        return out["coords_mut"][0].numpy()
    return out["coords"][0].numpy()


def _skeleton_atoms(base_atoms: dict, mut_seq: str, ca_coords: np.ndarray = None):
    """取 base 骨架原子（N/CA/C/O），CA 坐标可被 ca_coords 覆盖，残基名用 mut_seq。

    引擎 build 时用 AA 模板重建侧链 + 加氢（无需完整侧链）。
    """
    names, elems, seqs, resnames, coords = [], [], [], [], []
    cur = -1
    ca_idx = 0
    for i in range(len(base_atoms["res_seq"])):
        res = base_atoms["res_seq"][i]
        name = base_atoms["atom_names"][i]
        if res != cur:
            cur = res
            idx = min(res, len(mut_seq) - 1)
            new_aa = mut_seq[idx]
        if name in _BACKBONE:
            x, y, z = base_atoms["coords"][i]
            if name == "CA" and ca_coords is not None and ca_idx < len(ca_coords):
                x, y, z = ca_coords[ca_idx]
                ca_idx += 1
            names.append(name)
            elems.append(base_atoms["elements"][i])
            seqs.append(res)
            resnames.append(_AA3.get(new_aa, "ALA"))
            coords.append([x, y, z])
    return names, elems, seqs, resnames, np.asarray(coords, np.float32)


def build_mutant_structure(base_atoms: dict, mut_seq: str):
    """（回退）从原结构骨架 + 突变后序列重建 Structure。"""
    from spice_rl.env.structure import structure_from_atoms

    names, elems, seqs, resnames, coords = _skeleton_atoms(base_atoms, mut_seq)
    return structure_from_atoms(names, elems, seqs, resnames, coords)


def build_mutant_structure_from_ca(base_atoms: dict, mut_seq: str, pred_ca: np.ndarray):
    """用模型 Head B' 预测的突变后 Cα 坐标覆盖骨架 CA，重建 Structure。"""
    from spice_rl.env.structure import structure_from_atoms

    names, elems, seqs, resnames, coords = _skeleton_atoms(base_atoms, mut_seq, pred_ca)
    return structure_from_atoms(names, elems, seqs, resnames, coords)


def extract_base_atoms(structure) -> dict:
    """从引擎 Structure 读回原子数组（供突变重建用）。"""
    # 引擎未直接暴露原子数组；用 mmCIF 重建（或由调用方从 parquet 提供）。
    # 此处要求调用方直接传全原子 DataFrame/数组，见 path_b_search。
    raise NotImplementedError("请从数据管线直接提供全原子数组，见 path_b_search 参数")


# ---------------------------------------------------------------------------
# 路径 A：SAC 微观环（固定序列，双线程扰动环境）
# ---------------------------------------------------------------------------
def run_path_a(
    model,
    sac: SACTrainer,
    env: MDSimulationEnv,
    tokens, mask, z_mask,
    n_steps: int,
    dph: float, dT: float,
):
    """在扰动环境上跑 n_steps，收集经验并异步更新 SAC。

    双线程：sign=+1 / -1 分别扰动（+ΔpH/+ΔT 与 -ΔpH/-ΔT）。
    Returns: (crashed, env_fail, survive_steps)
    """
    state = env.state()
    crashed = False
    survive_steps = n_steps
    for step_idx in range(n_steps):
        z = encode_z(model, tokens, state["env"], mask)
        action_cont, action_disc = sac.act(
            z, state["env"], z_mask, deterministic=False
        )
        next_state, reward, done, info = env.step(action_cont)

        next_z = encode_z(model, tokens, next_state["env"], mask)
        sac.collect(
            {
                "z": z, "env": state["env"], "M": state["M"], "u_hist": state["u_hist"],
                "action_cont": action_cont, "action_disc": action_disc,
                "mutation_mask": float(info.get("mutation_allowed", 0.0)),
                "z_mask": z_mask,
                "reward": reward, "done": done,
                "next_z": next_z, "next_env": next_state["env"],
                "next_M": next_state["M"], "next_u_hist": next_state["u_hist"],
            }
        )
        state = next_state
        if done:
            crashed = bool(info["crashed"])
            survive_steps = step_idx + 1
            break
    env_fail = env.current_env() if crashed else None
    return crashed, env_fail, survive_steps


# ---------------------------------------------------------------------------
# 路径 B：ES 突变搜索 + 评估 + 伪标签回流
# ---------------------------------------------------------------------------
def path_b_search(
    model,
    sac,
    es,
    base_seq: str,
    env_fail,
    tokens, mask, z_mask,
    base_atoms,               # 全原子数组 dict（供突变后结构重建）
    pseudo_label_dir: str,
    survive_steps: int,
    env_cfg,
):
    """在 Env_fail 下用 ES 生成突变候选；冻结复用路径 A 的 SAC Actor 评估存活。

    突变后结构 = 模型 Head B' 输出的 Cα 坐标（覆盖骨架 CA）+ 引擎重建侧链。

    Returns: (survivors, best_seq, fitness, conf_samples)
      conf_samples: [(z_mut, [conf_A, conf_B])] 供 Head D 置信度训练
    """
    import spice_engine as se
    from spice_pre.data.preprocessing import seq_to_tokens

    os.makedirs(pseudo_label_dir, exist_ok=True)
    env_norm = _normalize_env(env_fail, env_cfg)

    candidates = es.propose_mutations(base_seq, tokens, env_norm, mask)

    survivors = []
    fitness = np.zeros(len(candidates), np.float32)
    conf_samples = []
    for j, (mut_seq, _k, _strategy) in enumerate(candidates):
        if mut_seq == base_seq or not _validate(mut_seq):
            fitness[j] = 0.0
            continue
        try:
            # 突变序列 + Env_fail 嵌入 z（冻结模型，仅前向）
            tok_mut = seq_to_tokens(mut_seq)
            Lm = tok_mut.shape[0]
            mask_mut = np.ones(Lm, np.float32)
            z_mut = encode_z(model, tok_mut, env_norm, mask_mut)
            z_mask_mut = np.zeros(z_mask.shape, np.float32)
            z_mask_mut[:Lm] = mask_mut
            # 突变后结构 = 模型 Head B' 输出坐标（覆盖骨架 CA）+ 引擎重建侧链
            pred_ca = predict_mutant_coords(model, tok_mut, env_norm, mask_mut)
            struct = build_mutant_structure_from_ca(base_atoms, mut_seq, pred_ca)
        except Exception as e:
            print(f"  突变 {mut_seq} 结构构建失败: {e}")
            fitness[j] = 0.0
            continue
        try:
            # 突变序列 + Env_fail 嵌入 z（冻结模型，仅前向）
            tok_mut = seq_to_tokens(mut_seq)
            Lm = tok_mut.shape[0]
            mask_mut = np.ones(Lm, np.float32)
            z_mut = encode_z(model, tok_mut, env_norm, mask_mut)
            z_mask_mut = np.zeros(z_mask.shape, np.float32)
            z_mask_mut[:Lm] = mask_mut

            eng = se.Engine.build(
                struct, float(env_fail[0]), float(env_fail[1]), 1.0, 0.0, 200, 2.0
            )
            # 冻结复用路径 A 的 SAC Actor 评估存活步数
            steps = 0
            for _ in range(survive_steps):
                a_cont, _ = sac.act(z_mut, env_norm, z_mask_mut, deterministic=False)
                out = eng.step(a_cont[: env_cfg.force_dim].astype(np.float32))
                steps += 1
                if out["crashed"]:
                    break
            fitness[j] = steps
            conf_b = min(1.0, steps / max(1, survive_steps))
            conf_samples.append((z_mut, np.array([0.0, conf_b], np.float32)))
            if steps >= survive_steps:
                pseudo = np.asarray(eng.pseudo_labels(), np.float32)
                fn = os.path.join(pseudo_label_dir, f"pseudo_{len(survivors)}_{steps}.npz")
                np.savez(fn, seq=mut_seq, env=np.array(env_fail), coords=pseudo)
                survivors.append((mut_seq, steps))
                print(f"  存活突变体: {mut_seq} (steps={steps}, {_strategy}) 伪标签 -> {fn}")
        except Exception as e:
            print(f"  突变评估失败: {e}")
            fitness[j] = 0.0
    best = max(survivors, key=lambda x: x[1], default=None)
    return survivors, (best[0] if best else base_seq), fitness, conf_samples


def _normalize_env(env_raw, env_cfg):
    """(ph, temp, ionic) → 归一化 [3] 环境向量。"""
    ph_n = (env_raw[0] - env_cfg.ph_min) / (env_cfg.ph_max - env_cfg.ph_min)
    t_n = (env_raw[1] - env_cfg.temp_min) / (env_cfg.temp_max - env_cfg.temp_min)
    i_n = float(np.clip(np.log10(max(env_raw[2], 1e-3)) / 3.0, 0.0, 1.0))
    return np.array([ph_n, t_n, i_n], np.float32)


def _validate(seq: str) -> bool:
    import spice_engine as se

    try:
        se.validate_sequence(seq)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def train(cfg: Config, structure, base_seq: str, base_atoms: dict = None):
    """运行双路后训练。

    structure: 初始野生型全原子 Structure（供引擎）
    base_seq:  野生型一字母序列
    base_atoms: 全原子数组 dict（路径 B 突变重建用；可为 None 则路径 B 禁用）
    """
    os.makedirs(cfg.post.log_dir, exist_ok=True)
    os.makedirs(cfg.post.ckpt_dir, exist_ok=True)

    # ---- 环节二：快筛（MD 短跑检查物理合法性，违规淘汰）----
    check = quick_check_env(
        structure, cfg.env, cfg.post.anchor_ph, cfg.post.anchor_temp
    )
    if not check["ok"]:
        print(f"快筛未通过（{check['reason']}），终止。请更换初始结构。")
        return
    print(f"快筛通过: U={check['u']:.1f} kcal/mol，进入双路探索")

    model = build_rl_model(cfg)
    L_max = cfg.sac.discrete_position_dim
    cont_dim = cfg.env.force_dim + cfg.env.env_offset_dim
    sac = SACTrainer(
        cfg.sac, z_dim=cfg_sac_z_dim(model), cont_dim=cont_dim, u_window=cfg.env.u_window
    )
    from spice_rl.es import ESEvolver

    es = ESEvolver(model, cfg.es)
    conf_trainer = ConfidenceHeadTrainer(model, lr=cfg.post.conf_lr)

    tokens, mask = tokens_from_seq(base_seq, L_max)
    z_mask = np.zeros(L_max, np.float32)
    z_mask[: len(mask)] = mask

    wt = base_seq
    start = time.time()
    for ep in range(cfg.post.max_episodes):
        # ---- 路径 A：固定当前序列，双线程扰动环境（+Δ / −Δ）----
        env_fail = None
        conf_a_steps = 0
        for sign in (+1, -1):
            env = MDSimulationEnv(
                structure, cfg.env,
                ph=cfg.post.anchor_ph + sign * cfg.post.env_delta_ph,
                temp=cfg.post.anchor_temp + sign * cfg.post.env_delta_T,
                ionic=cfg.env.ionic_default,
            )
            env.reset()
            crashed, fail, survive = run_path_a(
                model, sac, env, tokens, mask, z_mask,
                n_steps=cfg.env.episode_max_steps,
                dph=cfg.post.env_delta_ph, dT=cfg.post.env_delta_T,
            )
            conf_a_steps = max(conf_a_steps, survive)
            if crashed:
                env_fail = fail
                print(f"[ep {ep}] 路径A 崩溃，记录 Env_fail: {env_fail}")
                break
        # 收集路径 A 置信度样本（Head D 监督：存活步数/最大步数）
        z_wt = encode_z(
            model, tokens,
            _normalize_env(
                (cfg.post.anchor_ph, cfg.post.anchor_temp, cfg.env.ionic_default),
                cfg.env,
            ),
            mask,
        )
        conf_trainer.add(
            z_wt,
            np.array(
                [conf_a_steps / max(1, cfg.env.episode_max_steps), 0.0], np.float32
            ),
        )
        # 异步更新（收集满阈值时在 run_path_a 内已累计；这里再补一次更新）
        if len(sac.buffer) >= cfg.sac.batch_size:
            sac.update(z_mask)

        # ---- 路径 A：稳定性相图（探稳定区间/崩溃边界）----
        if cfg.post.phase_map_interval > 0 and ep % cfg.post.phase_map_interval == 0:
            try:
                pts = scan_phase_map(
                    structure,
                    cfg.post.phase_map_temp_range,
                    cfg.post.phase_map_ph_range,
                    pressure=cfg.env.pressure,
                    ionic=cfg.env.ionic_default,
                    relax_iters=cfg.env.relax_iters,
                    tolerance=cfg.env.tolerance,
                )
                s = summarize_phase_map(pts)
                out = os.path.join(cfg.post.phase_map_dir, f"phase_{ep:05d}.npz")
                save_phase_map(pts, out)
                print(
                    f"[ep {ep}] 相图: stable={s['n_stable']}/{s['n_points']} "
                    f"boundary={s['n_boundary']} crashed={s['n_crashed']} -> {out}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[ep {ep}] 相图扫描失败: {e}")

        # ---- 路径 B：若路径 A 崩溃，ES 启动突变搜索 ----
        if env_fail is not None and base_atoms is not None:
            survivors, wt, fitness, conf_samples = path_b_search(
                model, sac, es, wt, env_fail, tokens, mask, z_mask,
                base_atoms, cfg.post.pseudo_label_dir, cfg.es.fitness_survive_steps,
                cfg.env,
            )
            # 收集路径 B 置信度样本（供 Head D 训练）
            for z_m, c in conf_samples:
                conf_trainer.add(z_m, c)
            if survivors:
                # 存活突变体作为新亲本（序列更新，tokens 重建）
                tokens, mask = tokens_from_seq(wt, L_max)
                z_mask[: len(mask)] = mask
                # ---- 环节四：伪标签回流（写 TFRecord，供微调）----
                try:
                    write_pseudo_tfrecord(
                        cfg.post.pseudo_label_dir,
                        cfg.post.pseudo_tfrecord_path,
                        L_max,
                        weight_repeat=cfg.post.pseudo_weight_repeat,
                        survive_steps=cfg.es.fitness_survive_steps,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  伪标签回流失败: {e}")
            es.evolve(fitness)

        # Head D：双路置信度监督训练（存活步数/最大步数）
        if cfg.post.conf_train_interval > 0 and ep % cfg.post.conf_train_interval == 0:
            if len(conf_trainer) >= cfg.post.conf_batch:
                cl = conf_trainer.update(cfg.post.conf_batch)
                print(f"[ep {ep}] Head D 置信度 loss: {cl['conf_loss']:.4f}")

        if ep % cfg.post.log_every == 0:
            print(
                f"[ep {ep}] alpha={sac.alpha():.3f} buffer={len(sac.buffer)} "
                f"| wt={wt[:20]} | {time.time()-start:.0f}s", flush=True
            )


def cfg_sac_z_dim(model) -> int:
    # 从模型嵌入维度读取（Pre-train 的 embed_dim）
    return model.embed_dim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/posttrain.yaml")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--parquet-dir", default=None,
                    help="数据管线全原子 parquet 目录（生产入口）")
    ap.add_argument("--pdb-id", default=None,
                    help="parquet 中要加载的初始结构 pdb_id")
    ap.add_argument("--structure", default=None, help="(调试) mmCIF 路径")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_episodes is not None:
        cfg.post.max_episodes = args.max_episodes

    if args.parquet_dir and args.pdb_id:
        # 生产入口：数据管线全原子 → from_atoms（引擎自动加氢，无需 mmCIF）
        from spice_rl.env import load_structure_with_atoms

        struct, base_atoms = load_structure_with_atoms(args.parquet_dir, args.pdb_id)
        seq = struct.sequence()
        print(f"数据管线初始结构: {seq[:30]}... n_res={struct.residue_count()}")
        train(cfg, struct, seq, base_atoms=base_atoms)
    elif args.structure:
        # 调试入口：mmCIF 文件
        from spice_rl.env import structure_from_mmcif

        struct = structure_from_mmcif(args.structure)
        seq = struct.sequence()
        print(f"(调试) mmCIF 初始结构: {seq[:30]}... n_res={struct.residue_count()}")
        train(cfg, struct, seq)
    else:
        raise SystemExit(
            "需要 --parquet-dir + --pdb-id（数据管线）或 --structure（调试 mmCIF）"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
