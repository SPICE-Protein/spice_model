# Slurm 提交套件 —— 数据效率消融 coverage 轴（超算）

论文 §3.4 决定性实验：**固定 RL 环 × prior 尺度**。纯监督 prior 的 held-out 结果
（CASP14 AUC = 0.460 / 0.579 / 0.423 / 0.881）已在论文 + `data/data_efficiency/results.csv`；
覆盖轴要看**固定 RL 环**能不能让小 prior 也到同样的 beyond-PDB 区间。本套件就是给这个跑的。

## 文件

| 文件 | 作用 |
|---|---|
| `run_coverage_rl.sbatch` | 数组作业：`--array=10,100,1000,45000`，每个任务=一个尺度，同一组蛋白跑固定 RL 环 |
| `fetch_atoms.py` | 把选定蛋白的 atoms shard 从 HF 拉进本地 parquet 目录（幂等） |
| `proteins.txt` | （你填）每行一个 pdb_id，选好的蛋白组 |
| `setup_cluster.sh` | （集群登录节点跑）装 miniconda + 建 spice 环境(py3.12) + 依赖 + 引擎 wheel + 验证 |
| `pack_results.sh` | （集群跑）把各尺度 runs 打成 tar 供网页下载 |
| `UPLOAD.md` | 上传清单 + 布局 + 网页控制台操作 |

## 流程

**0. 生成各尺度 posttrain.yaml**（已在本地完成）
```bash
python scripts/experiments/ablate_data_efficiency.py --phase rl-gen \
    --post-config configs/posttrain.yaml --out ../data/data_efficiency
```

**1. 选定蛋白组**（关键）
- 用 Kaggle ④a/④b 那套筛：**margin 0.4–0.85**（2LYZ 0.86 是上限附近；3EMS 0.909 太稳→路径 B 空转 0 存活，别用）。
  也别用 margin≈0 的陷阱蛋白（只在极端 pH 崩、不可救）。
- **必须和 45k 全量 prior 那组蛋白一致** —— 否则小尺度对比没有共同分母。
- 建议 3–5 个蛋白（2k 核时预算见下）。把 pdb_id 填进 sbatch 的 `PROTEINS` 变量（或 `proteins.txt`）。

**3. 提交**
```bash
sbatch scripts/experiments/slurm/run_coverage_rl.sbatch
```
2 个任务并行（每尺度一个：n10 / n45000；中间尺度 100/1000 已剪，见下）。日志 `slurm_logs/cov_%A_%a.{out,err}`。

**4. 收集**：每尺度 `data/data_efficiency/n{N}/runs/posttrain/coverage.csv`
→ `scripts/experiments/ablate_physics_loop.py` 数 beyond-PDB 点 → 2 尺度 coverage 对比 → 填 §3.4 TODO。

## 核时预算（2026-08-14 裁剪版，目标：尽快出 Nature 投稿）

**裁剪决策**：coverage 轴只留 n10 + n45000 两尺度（主角=小 prior 能否到 beyond-PDB 覆盖，控制=大 prior）；
n100/n1000 的 held-out AUC 论文已有，RL 不必再跑。**D 消融（P3）与 FoldX/Rosetta 基线（P4）整体剪掉**
（后者是增量对比，优先级低，Nature 投稿可省）；**FireProtDB ΔΔG 放大实验已证无梯度（20 步窗是灾难过滤器非排序器），放弃**。

公式：`2 尺度 × K 蛋白 × C 核 × T 小时/蛋白 × 种子数`（2 尺度并行；墙钟 ≈ K×T×种子，核时 ≈ 2KCT×种子）

⚠️ **C=2 是稳妥默认**：引擎 SoA/SIMD 单线程、`path_a_threads` 是死配置（代码未引用），
单 run 吃 ~1 核 + TF 小头 → C=8 纯浪费 7/8。C=2 给 TF/SAC 留核即可。
T（单蛋白）按集群实测 ~2h（build≈2min/次 × 30 集）。

| 阶段 | K | C | T(h)/蛋白 | 种子 | 总核时 | 说明 |
|---|---|---|---|---|---|---|
| **P1 覆盖轴·验证管线** | 3 | 2 | 2 | 1 | **48** | 先跑通（n10+n45000），确认 coverage.csv 正常 |
| P2 覆盖轴·定稿 | 5 | 2 | 2 | 2 | 160 | K=5 + 2 种子，小尺度（弱种子）结论更稳 |
| ~~P3 D 消融~~ | - | - | - | - | **0** | ❌ 剪（Q-gate/conservation 作用已在 Methods §4.5.6 描述） |
| ~~P4 C 基线（FoldX/Rosetta）~~ | - | - | - | - | **0** | ❌ 剪（增量对比，Nature 投稿可省） |
| **合计** | | | | | **~210** | 2k 核时还剩 ~1790 缓冲 |

> 单蛋白 RL 墙钟：集群实测 build≈2min/次（溶剂初始化 ~37s + 最小化 75–100s），每集 ≥1 次 build + MD
> → 单蛋白粗估 ~2h，`--time = K × 2h × 1.5`。也可用 `--max-episodes 5` 冒烟一集实测再收紧。

## 注意事项

- **`--tag`**：sbatch 已传 `--tag $PDB`（`train_post.py` 新加的 CLI 参数），
  多蛋白的伪标签/候选表带蛋白前缀，不互相覆盖。用之前先确认上传了最新的 `train_post.py`。
- **相对路径**：per-scale yaml 里 `post.pretrain_ckpt` 等是相对路径，sbatch 已 `cd "$MODEL_ROOT"`，
  别从别的目录手动跑。
- **引擎线程**：脚本 `export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK`；但**先探针**（`probe_threads.py`）确认引擎是否吃核——SEv1.6+ 是 SoA/SIMD 单线程向量化、`path_a_threads` 是死配置，若引擎无 rayon，8 核=浪费 7 核。engine wheel 需在超算环境装好。
- **一个蛋白失败**：`train_post` 内部有 no_survivor_abort（默认 15 集），0 存活会抛错 → 脚本 `set -e` 中断。
  想跳过继续就删掉 `set -e` 或给 train_post 调用加 `|| echo "[warn] ${PDB} 失败，继续下一个"`。
- **零标签诚实性**：coverage.csv 是纯物理环证据（env_fail 点 = 引擎判定崩），
  B 消融读它数 beyond-PDB 点；别把"RL 探到了"和"物理就稳"混为一谈——图里分色看。
