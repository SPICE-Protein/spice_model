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
    source: str = "hf"                 
    hf_repo: str = "SPICE-Protein/spice_protein"
    hf_endpoint: str = "https://hf-mirror.com"  
    cache_dir: str = "data/cache"
    local_dir: str = "data/parquet"
    tfrecord_dir: str = "data/tfrecords"

    use_env_filtered: bool = True      
    min_seq_len: int = 40              
    max_seq_len: int = 512             
    max_shards: int = 0                
    structures_per_shard: int = 0      
    max_chains: int = 0                
    default_env: Tuple[float, float, float] = (7.0, 298.0, 0.15)  


@dataclass
class ModelConfig:
    vocab_size: int = 23    
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    env_dim: int = 3        
    pos_max_len: int = 1024
    dist_bins: int = 24     
    dist_dim: int = 64      
    dist_min: float = 3.0   
    dist_max: float = 48.0  
    bond_length: float = 3.8  # frame 结构头固定 Cα-Cα 虚拟键长（Å）
    distogram_fp16: bool = False  # fp16 跑 distogram 双线性 einsum（T4 tensor core ~8×；需混合精度）
    frame_recycle_steps: int = 2   # 🔴 recycling 精修轮数（2026-08-13）：治 frame 头 cumsum 长链累积；
                                   #   0=纯积分（旧行为），>0=距离感知 SE(3) 等变迭代纠偏（长链可学）
    frame_refine_dim: int = 64     # 精修模块注意力/MLP 隐维
    frame_refine_heads: int = 4    # 精修注意力头数


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 1000  
    grad_clip: float = 1.0
    val_split: float = 0.05
    seed: int = 42
    log_dir: str = "runs/pretrain"
    ckpt_dir: str = "checkpoints/pretrain"
    log_every: int = 50
    ckpt_every: int = 1000
    max_steps: int = 0     
    cache_train: bool = True        
    cache_path: str = ""           # 非空=落盘 cache（Dataset.cache(filename)，首次建、每 epoch 读文件，RAM 不囤）
    dist_weight: float = 1.0        
    pair_weight: float = 0.05       
    pair_warmup_steps: int = 3000   # frame 坐标损失权重从 0 线性升到 pair_weight（防随机游走梯度压制 CE）
    frame_chirality_weight: float = 1.0
    frame_clash_weight: float = 3.0  # 治自交；1.0 数值太弱（被 kabsch/consistency 淹没）→ Head A 仍有 ~100 对 <3Å 冲突，2026-08-14 拉高至 3.0 待验证
    frame_consistency_weight: float = 0.5  # 坐标↔distogram 自洽损失（全链，含长链，无需 native 标签；1.0 时 AUC -0.023 → 0.5 平衡）
    coord_max_len: int = 200    # 🔴 长度门控（2026-08-13 决定性）：frame 坐标损失只给 ≤coord_max_len 的链。
    # 🛑 平台期检测（2026-08-13）：val_dist 连续 plateau_patience 个 epoch 未显著改善 → 打 log 提示可停
    plateau_patience: int = 5       # 连续 N 个 epoch 无改善算平台期
    plateau_min_delta: float = 0.005  # 相对改善阈值（0.5%）：val_dist 需比 best 好这么多才算"有改善"
    plateau_auto_stop: bool = False   # True=平台期自动终止训练；False=只打 log（你看到后手动停）
                                #   长链（200-400aa 占全数据 65%）cumsum 误差累积→坐标炸→梯度毒 encoder→distogram 饿死。
                                #   distogram 全链学（长链距离可学，已验证），坐标监督只给短链。0=不门控
    use_gpu: bool = True            
    gpu_mem_growth: bool = True     
    gpu_devices: str = ""          
    use_mixed_precision: bool = False  


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

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
            if yaml is not None:
                d = yaml.safe_load(f) or {}
            else:
                d = json.load(f)
        if isinstance(d, dict):
            _apply_dict(cfg, d)
    return cfg
