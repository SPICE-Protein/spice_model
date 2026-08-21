from __future__ import annotations

import os
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ============ Global Eager Mode (2026-08-19, fallback switch for numerical instability in containerized TF graph mode) ============
# Set dynamically by SACTrainer(cfg.eager_update) / train(); module-level @tf.function blocks
# (such as train_post._encode_z_pool) read this flag to determine whether to execute via graph or eager mode.
_EAGER_MODE = False


def set_eager_mode(v: bool) -> None:
    global _EAGER_MODE
    _EAGER_MODE = bool(v)


def eager_mode() -> bool:
    return _EAGER_MODE


@dataclass
class EnvConfig:
    backend: str = "spice_engine"          
    engine_package: str = "spice_engine"   
    relax_iters: int = 200                 
    tolerance: float = 2.0                 
    strict_incomplete: bool = True         
    pressure: float = 0.0                  
    ionic_default: float = 0.0             
    episode_max_steps: int = 200           
    force_dim: int = 16                    
    force_clamp: float = 0.5               
    mutation_every: int = 20               
    env_offset_dim: int = 2                
    ph_rebuild_threshold: float = 0.05     
    env_offset_clamp: bool = True          
    env_dph_clamp: float = 2.0             
    env_dT_clamp: float = 20.0             
    env_abs_window: bool = True            
    env_ph_min: float = 2.0                
    env_ph_max: float = 10.0               
    env_temp_min: float = 260.0            
    env_temp_max: float = 330.0            
    u_window: int = 10                     
    metric_dim: int = 5                    
    ph_min: float = 0.0
    ph_max: float = 14.0
    temp_min: float = 150.0
    temp_max: float = 400.0
    reward_ref: float = 1e5              
    q_cutoff: float = 8.0                
    q_track: bool = True                 
    rmsf_window: int = 200               
    strain_reward_lambda: float = 0.0    
    strain_norm_ref: float = 10.0        
    strain_ratio_threshold: float = 2.0  
    m5_ratio_threshold: float = 1.3      


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005                     
    lr_actor: float = 3.0e-4
    lr_critic: float = 3.0e-4
    lr_alpha: float = 3.0e-4
    target_entropy_factor: float = 0.5
    buffer_capacity: int = 1_000_000       
    batch_size: int = 512
    update_every_steps: int = 20           # 2026-08-15: adjusted 200 -> 20. MD episodes average only ~20 steps;
                                           # a value of 200 would update SAC only once every ~10 episodes, leading to slow learning;
                                           # 20 ensures exactly one update per episode on average.
    update_iters: int = 1                  
    grad_clip: float = 1.0
    hidden_dim: int = 256
    u_ref: float = 1e5                  

    enable_mutation_head: bool = True      
    discrete_position_dim: int = 256       
    aa_dim: int = 20                       
    gumbel_tau: float = 1.0                

    track_mutation_mask: bool = True
    mutation_every: int = 20
    # 2026-08-19 Layer-by-layer NaN Tracing: when True, prints shape, finiteness, nbad, and absmax for each layer 
    # of the actor/critic, along with intermediate variables in _update_tf (q1, q2, y, log_pi) to locate where 
    # initial NaN values originate. Default is False (zero overhead in production); enable during 6QQE smoke tests on login nodes.
    trace_layers: bool = False
    # 2026-08-19 Eager Mode Fallback Switch: if containerized TF graph mode exhibits numerical issues (e.g., Dense forward 
    # activation explosion up to ~1e5, verifiable via smoke Check 6), setting this to True bypasses @tf.function optimizations 
    # for SAC updates and encode_z. Since the bottleneck is the physics engine (Rust MD), running RL in eager mode has negligible performance impact. Default is False.
    eager_update: bool = False


@dataclass
class ESConfig:
    population: int = 32                   
    elites: int = 8                        
    generations: int = 50
    mutation_rate: float = 0.1             
    crossover_rate: float = 0.5
    n_mutations_per_candidate: Tuple[int, int] = (1, 3)  
    head_b_noise: float = 0.05             
    head_c_noise: float = 0.05             
    freeze_backbone: bool = True           
    fitness_survive_steps: int = 200       
    conservation_mask: bool = True         
    conservation_threshold: float = 0.80   
    conservation_external: str = ""        
    q_gate: float = 0.50                   


@dataclass
class PostTrainConfig:
    max_episodes: int = 1000
    path_a_threads: int = 2                
    env_delta_ph: float = 0.5
    env_delta_T: float = 5.0
    anchor_ph: float = 7.0                 
    anchor_temp: float = 298.0
    max_seq_len: int = 150
    min_seq_len: int = 80                  
    reuse_engine: bool = True              
    quick_check_steps: int = 20            
    phase_map_dir: str = "runs/posttrain/phase_maps"
    phase_map_interval: int = 50           
    phase_map_temp_range: Tuple[float, float, float] = (250.0, 350.0, 10.0)
    phase_map_ph_range: Tuple[float, float, float] = (2.0, 11.0, 1.0)
    pseudo_label_dir: str = "data/pseudo_labels"   
    pseudo_tfrecord_path: str = "data/pseudo_labels/pseudo.tfrecord"  
    pseudo_weight_repeat: int = 8          
    pretrain_tfrecord_dir: str = "data/tfrecords"  
    finetune_epochs: int = 1
    finetune_batch_size: int = 16
    finetune_lr: float = 1.0e-4
    finetune_out: str = "checkpoints/pretrain/finetuned.weights.h5"
    conf_lr: float = 1.0e-4
    conf_batch: int = 16              
    conf_train_interval: int = 10       
    log_dir: str = "runs/posttrain"
    ckpt_dir: str = "checkpoints/posttrain"
    log_every: int = 10
    ckpt_every: int = 10
    no_survivor_abort: int = 15     
    # Env Escape & Recovery Mechanism (2026-08-18): prevents deadlocks when the agent pushes Path A into an unstable "basin of immediate collapse" 
    # (e.g., pH and temperature limits), leading to step-1 crashes. This causes the replay buffer to add only 1 transition per episode, 
    # stalling SAC updates and deadlocking training.
    # If the agent suffers "early collapse and 0 survivors" for `escape_after` consecutive episodes, training switches to a mild environment 
    # (anchor ± recovery_delta_*) with frozen environment offsets for `recovery_episodes` episodes. This feeds healthy transitions into 
    # the replay buffer, breaking the deadlock before resuming active environmental exploration.
    escape_step_threshold: int = 5     # 路径 A 存活 < 此值 计为"早崩"
    escape_after: int = 3              # 连续早崩集数 ≥ 此值 → 触发恢复
    recovery_episodes: int = 4         # 恢复期集数（温和 env + 冻结 env-offset）
    recovery_delta_ph: float = 1.0     # 恢复期 pH 偏移（anchor ± 1，稳定区）
    recovery_delta_T: float = 5.0      # 恢复期 temp 偏移（anchor ± 5）
    nan_watchdog: bool = True          # SAC 权重 NaN 看门狗：检测到非有限 → 重建 SAC 防死锁
    # Restart Gate: explosive mutation blacklisting (2026-08-16).
    # Identifies mutations that cause numerical explosions during equilibration or collapse during early simulation steps.
    # It assigns these mutations a light negative fitness bias so the ES learns that "explosion < 0 survival" while 
    # skipping redundant builds to save HPC core hours.
    restart_gate: bool = True          # 总开关
    explosive_threshold: int = 2       # 同一突变爆炸 ≥N 次 → 自动进黑名单
    explosive_min_steps: int = 3       # 运行早期崩溃判定：steps≤N 且 crashed = 爆炸
    explosive_penalty: float = -1.0    # 命中黑名单的 fitness（比 0 存活更糟）
    explosive_env_bucket: bool = True  # 黑名单按环境分桶（acid/neutral/base）：仅同桶崩溃累计/命中，防跨环境误伤
    explosive_seed: str = ""           # 可选种子黑名单文件（CSV: tag,pos,wt,mut,reason）；默认空=纯自动发现
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


def setup_logging(log_dir: str = "runs/posttrain", log_name: str = "train.log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    
    logger = logging.getLogger("spice")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Remove existing handlers to avoid duplication
    logger.handlers.clear()
    
    # File handler (detailed format including file and line number)
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    fh.setFormatter(fh_formatter)
    
    # Console handler (standardized clean format)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
