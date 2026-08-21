# 实验脚本（消融 + 基线）计划与用法

> 目标：把论文承诺的实验（§3.4 消融 + §3.3 基线 + physics-as-teacher 证据）都落成可跑脚本。
> 分工：**本地**跑模型/评估/小预训练；**超算或 Kaggle** 跑固定 RL 环与稳定性扫描（引擎 MD 是大头）。

## 概览

| 实验 | 脚本 | 论断 | 在哪跑 |
|---|---|---|---|
| A 数据效率消融 | `ablate_data_efficiency.py` | 物理供给覆盖、数据只当种子 | build/eval 本地；RL 环超算 |
| A' 消融结果报告 | `compare_coverage_axis.py` | n10 vs n45000 端点对比（存活/Q/coverage + Mann-Whitney + 图） | 本地 |
| B 物理环开关 | `ablate_physics_loop.py` | physics as teacher：on 有 beyond-PDB 点，off 恒 0 | 本地（读已有输出） |
| C 基线对比 | `benchmark_baselines.py` | SPICE 突变 vs FoldX/Rosetta/ΔΔG | 本地导出 + 超算跑工具 |
| D 功能约束消融 | （待写：Q-gate/conservation on/off 重跑路径 B） | 无约束推荐=存活但毁功能 | 超算/Kaggle |
| E 伪标签回流 | `ablate_reflow.py` | 物理回流教会折叠（Head A before/after） | 本地 |
| F 环境条件化消融 | （待写：AdaLN on/off） | pH/T 条件化是否真有用 | 本地 |

## A. 数据效率消融（§3.4，范式决定性检验）

```bash
# 1) 本地：各尺度建 TFRecord + 训 prior（max_chains 精确截断）
python scripts/experiments/ablate_data_efficiency.py --phase build --scales 10,100,1000,45000

# 2) 本地：各尺度在 CASP14 上算 held-out AUC
python scripts/experiments/ablate_data_efficiency.py --phase eval --scales 10,100,1000,45000

# 3) 超算/Kaggle：各尺度跑固定 RL 环（同一组蛋白）
python scripts/experiments/ablate_data_efficiency.py --phase rl-gen --scales 10,100,1000,45000
# → 打印每个尺度的 train_post 命令；RL 配置完全固定，只有 prior 尺度变
```

**关键纪律**：RL 环配置（同一组蛋白、同一 config）完全固定，只有 `post.pretrain_ckpt` 指向各尺度 checkpoint。

> ⚠️ 2026-08：消融定为**两点端点比较**（n10 vs n45000）。两个端点都是**当前 pretrain 管线**训的；
> 中间尺度（100/1000）只有旧 pretrain 的数字，与当前管线不可比，且被端点夹逼、不加信息，**不纳入消融**。

```bash
# 4) 本地：n10 vs n45000 端点消融报告（图 + Mann-Whitney + CSV）
python scripts/experiments/compare_coverage_axis.py \
    --base /path/to/seventh_mut --scales n10,n45000 \
    --labels 'n10=N=10;n45000=N=45,000'
# → 输出 runs/figures/coverage_axis_{base名}.png / .csv；每个 seed/round 换 --base 即可重跑
#   附带产物分析：runs/figures/product_chemistry_{base名}.png / .csv（存活突变体替换化学 + 两尺度共享位点），
#   控制台打印 top-N 替换与共享率——可量化"物理验证是 prior 无关、突变提议分布是 prior 相关"

## B. 物理环开关（physics as teacher 的 coverage 证据）

```bash
python scripts/experiments/ablate_physics_loop.py \
    --coverage runs/posttrain/coverage.csv --pdb-cloud data/entries_all.parquet
```
off（纯监督）恒 0 个 beyond-PDB 点（架构上界=数据分布）；on（完整环）统计探索点里 beyond-PDB 的个数。

## C. 基线对比

```bash
python scripts/experiments/benchmark_baselines.py --candidates runs/posttrain/pathb_candidates.csv
```
导出 `ready.csv`（唯一突变 + SPICE fitness/q），本机有 FoldX/Rosetta 就跑，没有就在超算跑，最后合成 `summary.csv` 画对比。

## E. 伪标签回流（Head A before/after）

```bash
python scripts/experiments/ablate_reflow.py \
    --before checkpoints/pretrain/best_weights.weights.h5 \
    --after  checkpoints/pretrain/finetuned.weights.h5
```

## 待写（需要引擎或模型改造）

- **D 功能约束消融**：`train_post` 加 `--no-q-gate` / `--no-conservation` 开关，重跑路径 B，对比 Q 分布。
- **F 环境条件化消融**：pretrain 加 env 恒置开关，对比条件化 vs 恒 env。
- 完成后在本 README 补命令。
