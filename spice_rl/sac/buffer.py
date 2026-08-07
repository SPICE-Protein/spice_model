"""全局 Replay Buffer（容量可配，默认 1e6，跨序列共享）。

每个 transition（定制 2/5 混合动作 + 分层掩码）：
    z            [D]          序列-环境嵌入（mean-pool）
    env          [3]          当前环境 (ph, T, ionic) 归一化
    M            [5]          五维物理指标
    u_hist       [W]          最近 W 步势能历史（Critic 特权输入）
    action_cont  [cont]       连续动作（偏置力[16]+环境偏移[2]）
    action_disc  [L_max+20]   离散突变 soft one-hot（位置 + 氨基酸）
    mutation_mask[1]          分层动作：该步是否允许突变
    z_mask       [L_max]      序列掩码（离散头 position 分布用，float16 省内存）
    reward       [1]
    next_z/next_env/next_M/next_u_hist   下一状态
    done         [1]

内存提示：容量 × 单条字节数（≈1.1k float32，L_max=256 时）≈ 4.4GB/1e6 条，
可按需调小 buffer_capacity。
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        z_dim: int,
        cont_dim: int,
        disc_dim: int,
        pos_dim: int,
        env_dim: int = 3,
        metric_dim: int = 5,
        u_window: int = 10,
    ):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.z = np.zeros((capacity, z_dim), np.float32)
        self.env = np.zeros((capacity, env_dim), np.float32)
        self.M = np.zeros((capacity, metric_dim), np.float32)
        self.u_hist = np.zeros((capacity, u_window), np.float32)
        self.action_cont = np.zeros((capacity, cont_dim), np.float32)
        self.action_disc = np.zeros((capacity, disc_dim), np.float32)
        self.mutation_mask = np.zeros((capacity, 1), np.float32)
        self.z_mask = np.zeros((capacity, pos_dim), np.float16)
        self.reward = np.zeros((capacity, 1), np.float32)
        self.next_z = np.zeros((capacity, z_dim), np.float32)
        self.next_env = np.zeros((capacity, env_dim), np.float32)
        self.next_M = np.zeros((capacity, metric_dim), np.float32)
        self.next_u_hist = np.zeros((capacity, u_window), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)

    def add(self, tr: dict) -> None:
        i = self.ptr
        self.z[i] = tr["z"]
        self.env[i] = tr["env"]
        self.M[i] = tr["M"]
        self.u_hist[i] = tr["u_hist"]
        self.action_cont[i] = tr["action_cont"]
        self.action_disc[i] = tr["action_disc"]
        self.mutation_mask[i, 0] = 1.0 if tr.get("mutation_mask", 1) else 0.0
        self.z_mask[i] = tr["z_mask"]
        self.reward[i, 0] = tr["reward"]
        self.next_z[i] = tr["next_z"]
        self.next_env[i] = tr["next_env"]
        self.next_M[i] = tr["next_M"]
        self.next_u_hist[i] = tr["next_u_hist"]
        self.done[i, 0] = 1.0 if tr["done"] else 0.0
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "z": self.z[idx],
            "env": self.env[idx],
            "M": self.M[idx],
            "u_hist": self.u_hist[idx],
            "action_cont": self.action_cont[idx],
            "action_disc": self.action_disc[idx],
            "mutation_mask": self.mutation_mask[idx],
            "z_mask": self.z_mask[idx].astype(np.float32),
            "reward": self.reward[idx],
            "next_z": self.next_z[idx],
            "next_env": self.next_env[idx],
            "next_M": self.next_M[idx],
            "next_u_hist": self.next_u_hist[idx],
            "done": self.done[idx],
        }

    def __len__(self) -> int:
        return self.size

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size
