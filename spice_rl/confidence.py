from __future__ import annotations

import collections
import random

import numpy as np
import tensorflow as tf


class ConfidenceHeadTrainer:

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
        self.buffer.append(
            (np.asarray(z, np.float32), np.asarray(conf, np.float32))
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def update(self, batch_size: int = 256) -> dict:
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
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.opt.apply_gradients(zip(grads, self.head_d.trainable_variables))
        return loss

    def predict(self, z: np.ndarray) -> np.ndarray:
        return self.head_d(
            tf.constant(np.asarray(z, np.float32)[None])
        ).numpy()[0]
