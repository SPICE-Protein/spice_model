from __future__ import annotations

import numpy as np
import tensorflow as tf

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _evolvable_vars(model, policy_head):
    out = []
    for head_name in ("head_b", "head_c"):
        head = getattr(model, head_name, None)
        if head is not None and isinstance(head, tf.keras.layers.Layer):
            for var in head.trainable_variables:
                out.append((f"{head_name}/{var.name}", var))
    if policy_head is not None:
        for li, layer in enumerate(policy_head.layers):
            for var in layer.trainable_variables:
                out.append((f"policy/l{li}/{var.name}", var))
    if not out:
        raise ValueError(
            "模型未启用 Head B/C。请用 SPICEPretrainModel(cfg, heads=('A','B','Bp','C','D')) 构建。"
        )
    return out


class ESEvolver:
    def __init__(self, model, cfg, seed: int = 0):
        self.model = model
        self.cfg = cfg
        self.policy_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu", name="es_policy_d0"),
                tf.keras.layers.Dense(2, name="es_policy_out"),
            ]
        )
        z_dim = getattr(model, "embed_dim", 256)
        self.policy_head(tf.zeros([1, z_dim], tf.float32))
        self.head_keys, self.head_vars = zip(*_evolvable_vars(model, self.policy_head))
        self.head_keys = list(self.head_keys)
        self.head_vars = list(self.head_vars)
        self.base = [tf.identity(v) for v in self.head_vars]
        self.base_shapes = [v.shape.as_list() for v in self.head_vars]
        self.rng = np.random.default_rng(seed)
        self.noise = None

    def _init_population(self):
        pop = []
        for _ in range(self.cfg.population):
            ind = {
                self.head_keys[i]: self.rng.normal(
                    0.0, self.cfg.head_b_noise, size=tuple(self.base_shapes[i])
                ).astype(np.float32)
                for i in range(len(self.head_keys))
            }
            pop.append(ind)
        self.noise = pop

    def _set_weights(self, individual: dict):
        for i, v in enumerate(self.head_vars):
            v.assign(self.base[i] + individual[self.head_keys[i]])

    def _restore_base(self):
        for i, v in enumerate(self.head_vars):
            v.assign(self.base[i])

    def _policy(self, z_pool: np.ndarray) -> tuple:
        logits = self.policy_head(tf.constant(np.asarray(z_pool, np.float32)[None])).numpy()[0]
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        if self.rng.random() < p[0]:
            return (1, 1), "conservative"
        return self.cfg.n_mutations_per_candidate, "aggressive"

    def propose_mutations(
        self, seq: str, tokens: np.ndarray, env: np.ndarray, mask: np.ndarray
    ) -> list:
        if self.noise is None:
            self._init_population()

        candidates = []
        z_pool = self._z_pool(tokens, env, mask)
        for ind in self.noise[: self.cfg.population]:
            self._set_weights(ind)
            logits = self._head_b_logits(tokens, env, mask)   
            n_mut_range, strategy = self._policy(z_pool)
            mut_seq, k = self._sample_from_logits(seq, logits, n_mut_range, strategy)
            candidates.append((mut_seq, k, strategy))
        self._restore_base()
        return candidates

    def _z_pool(self, tokens, env, mask) -> np.ndarray:
        inp = {
            "tokens": tf.constant(tokens.astype(np.int32)[None]),
            "env": tf.constant(np.asarray(env, np.float32)[None]),
            "mask": tf.constant(np.asarray(mask, np.float32)[None]),
        }
        out = self.model(inp, training=False)
        z = out["z"][0]
        m = tf.constant(np.asarray(mask, np.float32))[:, None]   
        return (tf.reduce_sum(z * m, axis=0) / tf.maximum(tf.reduce_sum(m), 1.0)).numpy()

    def _head_b_logits(self, tokens, env, mask) -> np.ndarray:
        inp = {
            "tokens": tf.constant(tokens.astype(np.int32)[None]),
            "env": tf.constant(np.asarray(env, np.float32)[None]),
            "mask": tf.constant(np.asarray(mask, np.float32)[None]),
        }
        out = self.model(inp, training=False)
        logits = out["mutation"][0].numpy()          
        logits[~np.asarray(mask, bool)] = -1e9       
        return logits

    def _sample_from_logits(self, seq, logits, n_mut_range, strategy):
        L, n_aa = logits.shape
        if strategy == "conservative":
            n_mut = 1
        else:
            n_mut = int(self.rng.integers(n_mut_range[0], n_mut_range[1] + 1))
        mut_seq = seq
        k = 0
        if strategy == "conservative":
            pos = [int(np.argmax(logits.max(axis=1)))]
        else:
            pos = np.argsort(-logits.max(axis=1))[: max(n_mut, 3)]
            self.rng.shuffle(pos)
            pos = pos[:n_mut]
        for p in pos:
            lp = np.asarray(logits[p], dtype=np.float64)
            if not np.all(np.isfinite(lp)):
                continue
            probs = np.exp(lp - lp.max())
            probs = probs / probs.sum()
            if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
                continue
            to = int(self.rng.choice(n_aa, p=probs))
            to_aa = AA20[to]
            if to_aa == mut_seq[p]:
                continue
            new_seq = self._mutate_one(mut_seq, int(p), to_aa)
            if new_seq is not None and new_seq != mut_seq:
                mut_seq = new_seq
                k += 1
        return mut_seq, k

    @staticmethod
    def _mutate_one(seq, p, to_aa):
        try:
            import spice_engine as se

            return se.mutate_sequence(seq, int(p), to_aa)
        except Exception:  # noqa: BLE001
            if 0 <= p < len(seq):
                return seq[:p] + to_aa + seq[p + 1 :]
            return seq

    @staticmethod
    def _valid(seq) -> bool:
        try:
            import spice_engine as se

            se.validate_sequence(seq)
            return True
        except Exception:  # noqa: BLE001
            return all(c in AA20 for c in seq)

    def evolve(self, fitness: np.ndarray) -> None:
        if self.noise is None:
            self._init_population()
        order = np.argsort(-np.asarray(fitness))
        elites = [self.noise[i] for i in order[: self.cfg.elites]]
        new_pop = list(elites)
        n_elites = len(elites)
        while len(new_pop) < self.cfg.population:
            parent = elites[self.rng.integers(0, n_elites)]
            child = self._mutate_individual(parent)
            if self.cfg.crossover_rate > 0.0 and n_elites > 1:
                if self.rng.random() < self.cfg.crossover_rate:
                    other = elites[self.rng.integers(0, n_elites)]
                    child = self._crossover(child, other)
            new_pop.append(child)
        self.noise = new_pop[: self.cfg.population]

    def _mutate_individual(self, ind: dict) -> dict:
        child = {}
        for i, key in enumerate(self.head_keys):
            shape = self.base_shapes[i]
            child[key] = ind[key] + self.rng.normal(
                0.0, self.cfg.head_b_noise, size=shape
            ).astype(np.float32)
        return child

    def _crossover(self, a: dict, b: dict) -> dict:
        child = {}
        for i, key in enumerate(self.head_keys):
            shape = self.base_shapes[i]
            m = self.rng.random(shape) < 0.5
            child[key] = np.where(m, a[key], b[key]).astype(np.float32)
        return child
