"""Head D（双路置信度头）监督训练。

文档：Head D 以"该环境下 MD 存活步数 / 最大步数"为监督标签，
学习预测两条路径（A/B）的成功概率 [0,1]^2。

- 标签 conf_A：路径 A（固定序列）在该环境的存活率
- 标签 conf_B：路径 B（突变体）在该环境的存活率

样本收集于 train_post 双路循环，这里用 MSE 回归，只更新 model.head_d。
"""
from __future__ import annotations

import collections
import random

import numpy as np
import tensorflow as tf


class ConfidenceHeadTrainer:
    """Head D 置信度回归训练器。"""

    def __init__(self, model, lr: float = 1e-4, buffer_capacity: int = 20000):
        self.model = model
        self.head_d = getattr(model, "head_d", None)
        if self.head_d is None:
            raise ValueError(
                "模型未启用 Head D。请用 SPICEPretrainModel(cfg, heads=('A','B','Bp','C','D')) 构建。"
            )
        self.opt = tf.keras.optimizers.Adam(lr)
        self.buffer = collections.deque(maxlen=buffer_capacity)

    def add(self, z: np.ndarray, conf: np.ndarray) -> None:
        """收集样本。z [D]，conf [2]（conf_A, conf_B）。"""
        self.buffer.append(
            (np.asarray(z, np.float32), np.asarray(conf, np.float32))
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def update(self, batch_size: int = 256) -> dict:
        """从样本缓冲采样 batch 训练 Head D。"""
        if len(self.buffer) < max(batch_size, 16):
            return {"conf_loss": float("nan")}
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        z = np.stack([b[0] for b in batch])
        conf = np.stack([b[1] for b in batch])
        loss = self._update_tf(tf.constant(z), tf.constant(conf))
        return {"conf_loss": float(loss)}

    @tf.function
    def _update_tf(self, z, conf):
        with tf.GradientTape() as tape:
            pred = self.head_d(z)
            loss = tf.reduce_mean(tf.square(pred - conf))
        grads = tape.gradient(loss, self.head_d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.head_d.trainable_variables))
        return loss

    def predict(self, z: np.ndarray) -> np.ndarray:
        """预测 [conf_A, conf_B]。"""
        return self.head_d(
            tf.constant(np.asarray(z, np.float32)[None])
        ).numpy()[0]
