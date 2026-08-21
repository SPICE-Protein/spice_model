# SPICE Pre-train Model — ONNX Inference Guide

This document describes the exported ONNX model of the SPICE **Phase-1 (pre-train)** model, how to use it for high-speed inference (e.g. from a Rust GUI via the [`ort`](https://github.com/pykeio/ort) crate), and the exact I/O + preprocessing contract.

- **Model file:** [`export/spice_infer_fixed.onnx`](../export/spice_infer_fixed.onnx)
- **Export script:** [`scripts/export_onnx.py`](../scripts/export_onnx.py)
- **Version:** opset 13, fp32, weights = released pre-train checkpoint (`checkpoints/pretrain/best_weights.weights.h5`)

---

## 1. What is exported

The ONNX graph is the **frozen inference graph** of the pre-trained SPICE model, exporting **all heads** so the inference contract is stable from pre-train through RL:

- Inputs: tokenized sequence + normalized environment + attention mask
- Outputs: **Cα coordinates** (Head A), **distogram** (pairwise distance distribution), **mutation logits** (Head B), **mutant coordinates** (Head B'), **environment offset** (Head C), and **confidence** (Head D)

Two notes on correctness:

- **fp32.** The exported graph computes the distogram in fp32. During training the bilinear distogram einsum runs in fp16 purely as a T4 compute optimization; the fp32 graph uses the same weights and is mathematically equivalent to the trained model.
- **Custom ops are already rewritten.** `tf2onnx` cannot map two TF ops used by the model; the export script rewrites them in the graph so it is standard ONNX:
  - GELU's `Erfc` → `1 − Erf(x)` (10 occurrences, in the Transformer FFN and Head A MLPs)
  - `Cross` (`tf.linalg.cross`) → component-wise `Mul/Sub/Gather` decomposition (3 occurrences, in the frame decoder and the SE(3) recycling module)

> **Head-weight availability (important).** The released **pre-train** checkpoint trains only Head A + the distogram. In this exported model, **Head B / B' / C / D have random (untrained) weights** — running them now returns meaningless values (Head B' in particular is a known limitation). Those heads are trained in the **RL (post-train)** stage; when the RL checkpoint is released, **re-run `scripts/export_onnx.py`** to regenerate the ONNX from it — the I/O contract below is unchanged, only the weights improve. The GUI can therefore be built now against the full contract.

---

## 2. I/O contract

### Inputs

| Name | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `tokens` | `[B, L]` | `int32` | Tokenized sequence: `0` = padding, `1..20` = amino acids, `21` = unknown |
| `env` | `[B, 3]` | `float32` | Normalized environment `[pH, temperature, ionic_strength]` |
| `mask` | `[B, L]` | `float32` | `1.0` = valid residue, `0.0` = padding |

**`L` is dynamic** — the model is fully variable-length. The training corpus spans 40–512 residues; the positional encoding caps at 1024. `onnxruntime` / `ort` handle dynamic dimensions natively; for a GUI each protein carries its own `L`, or you may pad to a fixed length for batched requests.

### Outputs

| Name | Shape | Dtype | Description | Head / status |
| :--- | :--- | :--- | :--- | :--- |
| `coords` | `[B, L, 3]` | `float32` | Predicted Cα coordinates (Å), one point per residue | Head A — trained |
| `dist_logits` | `[B, L, L, 48]` | `float32` | Symmetrized logits over 48 distance bins (3–48 Å) | distogram head — trained |
| `mutation` | `[B, L, 20]` | `float32` | Per-position mutation logits over 20 amino acids (softmax for probabilities) | Head B — RL-only, random here |
| `coords_mut` | `[B, L, 3]` | `float32` | Mutant Cα coordinates (Å) = Head A fold of the (mutant) sequence (alias of `coords`) | Head B' — alias of Head A |
| `env_offset` | `[B, 2]` | `float32` | Active environment offset `[ΔpH, ΔT]` | Head C — RL-only, random here |
| `conf` | `[B, 2]` | `float32` | Two-path stability confidence `[0, 1]²` (sigmoid) | Head D — RL-only, random here |

---

## 3. Preprocessing (must match exactly)

The input tensors must be produced with the same conventions as the Python pipeline (`spice_pre/data/preprocessing.py`).

### 3.1 Tokenization

One-letter amino acids map to `1..20` in this order:

```
A C D E F G H I K L M N P Q R S T V W Y
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
```

Any other character (e.g. `X`, `B`, `Z`, `U`, `O`) → `21` (unknown). Padding → `0`.

### 3.2 Environment normalization

All three components are **already normalized to ~[0, 1]** by the caller:

| Component | Formula | Default |
| :--- | :--- | :--- |
| pH | `clip((pH − 0) / (14 − 0), 0, 1)` | 7.0 → 0.5 |
| Temperature (K) | `clip((T − 150) / (400 − 150), 0, 1)` | 298 K |
| Ionic strength (M) | `clip(log10(ionic / 1e-3) / log10(1e3), 0, 1)`, ionic first clipped to `[1e-3, 1.0]` | 0.15 M |

### 3.3 Mask

`mask[i] = 1` for every real residue, `0` for padding. Must be consistent with `tokens` (padding positions are masked out).

---

## 4. Postprocessing

### 4.1 Coordinates

`coords` is the Cα trace in Å (absolute scale) — use it directly for 3D visualization (polyline / tube rendering), optionally centered on the centroid.

> **Interpretation caveat.** The pre-trained coordinates are a **coarse prior** ("clay"), not high-accuracy folded structures; refinement to clean conformations is the physics/RL stage's job. Do not interpret them as AlphaFold-level accuracy, especially for long chains.

### 4.2 Distance map / contacts

`dist_logits` are logits over 48 bins spanning 3–48 Å. Bin edges are `linspace(3, 48, 47)`; the first bin center ≈ 1.5 Å, the last ≈ 60 Å.

Contact probability (Cα–Cα < 8 Å):

```python
import numpy as np
p = softmax(dist_logits, axis=-1)                  # [B, L, L, 48]
edges = np.linspace(3.0, 48.0, 47)
p_contact = p[..., edges < 8.0].sum(axis=-1)       # [B, L, L]
```

Expected pairwise distances:

```python
centers = np.concatenate([[0.0], edges, [72.0]])
centers = (centers[:-1] + centers[1:]) / 2.0
d_exp = (p * centers).sum(axis=-1)                 # [B, L, L]
```

### 4.3 Mutation / confidence / env-offset heads (RL stage)

- `mutation` → softmax over the last axis gives per-position mutation probabilities `[B, L, 20]`.
- `conf` is already in `[0, 1]` (sigmoid): `[pathA_confidence, pathB_confidence]`.
- `env_offset` is a raw `[ΔpH, ΔT]` suggestion (not normalized).

Until the RL checkpoint is exported, these three are untrained (random) and should be shown as "not available" in the GUI.

---

## 5. Running inference

### Python (onnxruntime)

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("export/spice_infer_fixed.onnx",
                            providers=["CPUExecutionProvider"])
out = sess.run(None, {"tokens": tokens, "env": env, "mask": mask})
coords, dist_logits = out[0], out[1]   # [B, L, 3], [B, L, L, 48]
```

### Rust (`ort` crate)

```rust
use ort::session::Session;

let session = Session::builder()?
    .commit_from_file("export/spice_infer_fixed.onnx")?;

let outputs = session.run(ort::inputs! {
    "tokens" => tokens_tensor,   // int32 [1, L]
    "env"    => env_tensor,      // f32   [1, 3]
    "mask"   => mask_tensor,     // f32   [1, L]
}?)?;
```

(`ort::inputs!` / tensor construction details depend on the `ort` version — see the [ort docs](https://github.com/pykeio/ort).)

---

## 6. Regenerating the ONNX

Run from the repository root:

```bash
PYTHONPATH=. python scripts/export_onnx.py
```

The script:
1. Loads `configs/pretrain.yaml` and the checkpoint `checkpoints/pretrain/best_weights.weights.h5`.
2. Builds the model with `heads=("A",)` and forces the fp32 distogram path.
3. Wraps it in a `tf.function` with a **dynamic-length** input signature.
4. Converts with `tf2onnx` (opset 13).
5. Rewrites `Erfc` → `1 − Erf` and `Cross` → component decomposition.
6. Writes `export/spice_infer_fixed.onnx`.

Dependencies: `tensorflow`, `tf2onnx`, `onnx`, `onnxruntime` (pip-installable).

### Correctness (parity check)

The export script runs a numerical parity check (Keras vs onnxruntime) on random proteins. All six outputs match (max abs diff):

| L | coords | dist_logits (rel) | mutation | coords_mut | env_offset | conf |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 60 | 1.2e-05 | 3.7e-04 (6e-07) | 7.2e-06 | 5.2e-06 | 7.8e-07 | 1.8e-07 |
| 150 | 9.7e-06 | 4.3e-04 | 5.5e-06 | 5.3e-06 | 1.3e-06 | 2.4e-07 |
| 300 | 1.5e-04 | 4.3e-04 | 5.7e-06 | 5.1e-06 | 1.4e-06 | 2.7e-07 |

Differences are float32 rounding noise; the ONNX graph is numerically equivalent to the Keras model.

---

## 7. Constraints & notes

- **Dynamic `L`**: trained on 40–512 residues; `L` up to 1024 works (positional-encoding cap). The distogram is `O(L² · 48)` — for `L = 512` one sample is ~50 MB fp32, so keep `L` modest in a GUI (150–300 is typical).
- **fp32 only**: no fp16 providers needed; export forces the fp32 einsum path.
- **Acceleration**: ONNX Runtime CPU is already fast for single proteins; CUDA / CoreML providers work out of the box if available.
- **Minimum runtime**: opset 13 ⇒ any ONNX Runtime ≥ 1.12.
