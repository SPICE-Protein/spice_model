"""RL（Post-train）配置：环境 / SAC / ES / 双路循环。

YAML 覆盖方式与 Pre-train 的 Config 一致（load_config）。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class EnvConfig:
    # spice_engine 后端
    backend: str = "spice_engine"          # 引擎模块名（PyO3 编译产物）
    engine_package: str = "spice_engine"   # import 名
    relax_iters: int = 200                 # Engine.build 的弛豫迭代
    tolerance: float = 2.0                 # 能量最小化容差 (kcal/mol)
    pressure: float = 1.0                  # bar
    ionic_default: float = 0.0             # M（引擎默认 0）
    # 每幕
    episode_max_steps: int = 200           # 单幕最大 MD 步（超过自然结束，R_term=0）
    # 动作空间（与引擎 ForceAction 一致）
    force_dim: int = 16                    # M：偏置力基元数
    force_clamp: float = 0.5               # kcal/(mol·Å)
    mutation_every: int = 20               # 分层动作：突变每 20 步
    env_offset_dim: int = 2                # [ΔpH, ΔT]
    # 状态
    u_window: int = 10                     # 历史势能窗口（Critic 特权输入）
    metric_dim: int = 5                    # M = [m1..m5]
    # 环境边界
    ph_min: float = 0.0
    ph_max: float = 14.0
    temp_min: float = 150.0
    temp_max: float = 400.0
    reward_scale: float = 1.0              # 1.0 = 严格 -U_t（kJ/mol）


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005                     # 软更新
    lr_actor: float = 3.0e-4
    lr_critic: float = 3.0e-4
    lr_alpha: float = 3.0e-4
    # 自适应熵：目标熵 = -dim * factor（文档：默认值的一半）
    target_entropy_factor: float = 0.5
    buffer_capacity: int = 1_000_000       # 全局 Replay Buffer
    batch_size: int = 512
    update_every_steps: int = 500          # 每收集满 N 步更新一次（异步收集）
    update_iters: int = 1                  # 每次更新迭代轮数
    grad_clip: float = 1.0
    hidden_dim: int = 256

    # 定制 2：混合动作解耦输出头
    enable_mutation_head: bool = True      # 离散突变头（Gumbel-Softmax τ）
    discrete_position_dim: int = 256       # 突变位置 one-hot 维度（= 最长序列，buffer 固定）
    aa_dim: int = 20                       # 目标氨基酸类型数
    gumbel_tau: float = 1.0                # Gumbel-Softmax 温度

    # 定制 5：分层动作时序（突变每 mutation_every 步生效一次）
    track_mutation_mask: bool = True
    mutation_every: int = 20


@dataclass
class ESConfig:
    population: int = 32                   # 每批候选突变体
    elites: int = 8                        # 保留精英
    generations: int = 50
    mutation_rate: float = 0.1             # 每个位点突变概率
    crossover_rate: float = 0.5
    n_mutations_per_candidate: Tuple[int, int] = (1, 3)  # 每候选 1~3 个点突变
    head_b_noise: float = 0.05             # Head-B 权重扰动幅度
    head_c_noise: float = 0.05             # Head-C 权重扰动幅度
    freeze_backbone: bool = True           # 冻结 Transformer 主干
    fitness_survive_steps: int = 200       # 适应度：存活步数（超过即稳定）


@dataclass
class PostTrainConfig:
    max_episodes: int = 1000
    path_a_threads: int = 2                # 路径 A 双线程扰动（+Δ / −Δ）
    env_delta_ph: float = 0.5
    env_delta_T: float = 5.0
    anchor_ph: float = 7.0                 # 起始环境（野生型默认）
    anchor_temp: float = 298.0
    # 环节二：快筛
    quick_check_steps: int = 20            # 快筛短跑步数（物理合法性检查）
    # 路径 A：稳定性相图（探稳定区间/崩溃边界）
    phase_map_dir: str = "runs/posttrain/phase_maps"
    phase_map_interval: int = 50           # 每 N episode 扫一次相图（0=关闭）
    phase_map_temp_range: Tuple[float, float, float] = (250.0, 350.0, 10.0)
    phase_map_ph_range: Tuple[float, float, float] = (2.0, 11.0, 1.0)
    # 环节四：伪标签回流
    pseudo_label_dir: str = "data/pseudo_labels"   # 伪标签 npz 落盘目录
    pseudo_tfrecord_path: str = "data/pseudo_labels/pseudo.tfrecord"  # 回流 TFRecord
    pseudo_weight_repeat: int = 8          # 置信度权重重复系数
    pretrain_tfrecord_dir: str = "data/tfrecords"  # Pre-train 原 TFRecord
    # 微调
    finetune_epochs: int = 1
    finetune_batch_size: int = 16
    finetune_lr: float = 1.0e-4
    finetune_out: str = "checkpoints/pretrain/finetuned.weights.h5"
    # Head D：双路置信度监督训练（标签 = 存活步数/最大步数）
    conf_lr: float = 1.0e-4
    conf_batch: int = 256
    conf_train_interval: int = 10       # 每 N episode 训练一次 Head D（0=关闭）
    log_dir: str = "runs/posttrain"
    ckpt_dir: str = "checkpoints/posttrain"
    log_every: int = 10
    ckpt_every: int = 100
    pretrain_ckpt: str = "checkpoints/pretrain/best_weights.weights.h5"
    pretrain_config: str = "configs/pretrain.yaml"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    es: ESConfig = field(default_factory=ESConfig)
    post: PostTrainConfig = field(default_factory=PostTrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _apply_dict(dc: Any, d: Dict[str, Any]) -> None:
    for k, v in d.items():
        if not hasattr(dc, k):
            continue
        cur = getattr(dc, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _apply_dict(cur, v)
        else:
            setattr(dc, k, v)


def load_config(path: Optional[str] = None) -> Config:
    cfg = Config()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) if yaml is not None else {}
        if isinstance(d, dict):
            _apply_dict(cfg, d)
    return cfg
