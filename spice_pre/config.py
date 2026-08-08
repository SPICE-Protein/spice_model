"""集中式配置管理。

支持两种方式：
1. 纯 Python：`Config()` 提供全部默认值。
2. YAML 覆盖：`load_config("configs/pretrain.yaml")` 用文件里的键覆盖默认值。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class DataConfig:
    # 数据来源
    source: str = "hf"                 # "hf" 从 HuggingFace | "local" 读本地 parquet 目录
    hf_repo: str = "SPICE-Protein/spice_protein"
    hf_endpoint: str = "https://hf-mirror.com"  # 本地开发用国内镜像；Colab 自动切官方端点
    cache_dir: str = "data/cache"
    local_dir: str = "data/parquet"
    tfrecord_dir: str = "data/tfrecords"

    # 清洗过滤
    use_env_filtered: bool = True      # 仅保留 has_env=True（设计文档：只训带环境标签的 5.6%）
    min_seq_len: int = 40              # 最短序列
    max_seq_len: int = 512             # 最长序列（超长截断 / 训练 padding 长度）
    max_shards: int = 0                # 0 = 全部 shard（调试可设小值）
    structures_per_shard: int = 0      # 0 = 每 shard 全部结构（调试可设小值）
    default_env: Tuple[float, float, float] = (7.0, 298.0, 0.15)  # pH, T, ionic(M)


@dataclass
class ModelConfig:
    vocab_size: int = 23    # PAD(0) + 20 AA + UNK(21)
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    env_dim: int = 3        # [pH, T, ionic] 归一化后
    pos_max_len: int = 1024
    dist_bins: int = 24     # binned distogram 距离分箱数（0 或 1 表示关掉）；64 箱方案内存不够，折中 24
    dist_dim: int = 64      # distogram 头投影维度（因子化双线性的中间维度）
    dist_min: float = 3.0   # distogram 最小距离（Å）
    dist_max: float = 48.0  # distogram 最大距离（Å），超出落溢出 bin


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 1000  # 线性 warmup（yaml 可覆盖）
    grad_clip: float = 1.0
    val_split: float = 0.05
    seed: int = 42
    log_dir: str = "runs/pretrain"
    ckpt_dir: str = "checkpoints/pretrain"
    log_every: int = 50
    ckpt_every: int = 1000
    max_steps: int = 0     # 0 = 按 epochs 跑满
    cache_train: bool = True        # True = 把整个 epoch 的 batch 预存内存，管线不再拖累计算
    dist_weight: float = 1.0        # binned distogram CE 权重（主目标）
    pair_weight: float = 0.05       # 坐标两两距离 RMSE 辅助权重（0.3→0.05：坐标头噪声梯度干扰 CE 主导，已下调）
    use_gpu: bool = True            # True = 用 GPU（若有）；False = 强制只用 CPU
    gpu_mem_growth: bool = True     # 显存按需增长（避免一次性占满）
    gpu_devices: str = ""          # GPU 编号白名单，如 "0" / "0,1"；空 = 全部
    use_mixed_precision: bool = False  # True = mixed_float16（T4 提速 ~2-4x；loss 内 SVD 保持 fp32）


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _apply_dict(dc: Any, d: Dict[str, Any]) -> None:
    """递归地把 dict 中的键覆盖到 dataclass 上（忽略未知键）。"""
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
            if yaml is not None:
                d = yaml.safe_load(f) or {}
            else:
                d = json.load(f)
        if isinstance(d, dict):
            _apply_dict(cfg, d)
    return cfg
