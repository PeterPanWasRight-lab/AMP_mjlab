# AMP 专家数据集对照实验

本目录包含 3 个对照实验的启动脚本和完整教程文档，用于理解 AMP 训练中专家数据集如何驱动 G1 机器人。

## 快速开始

```bash
# 三个对照实验（按推荐顺序）
bash exp_C_passthrough.sh        # 纯展示 npz 形态（默认 60 秒循环）
bash exp_A_pd.sh                 # PD 跟踪 + 真物理（默认 30 秒循环）
bash exp_B_pd_target_zero.sh     # 对照组（不接 npz，默认 30 秒）
```

## 参数说明

每个脚本接受两个位置参数：
- **第 1 个**：npz 文件路径
- **第 2 个**：运行时长（秒），`0` 表示无限

```bash
# 看某个特定数据集 10 秒
bash exp_C_passthrough.sh src/assets/motions/g1/amp/WalkandRun/jog_forward_loop_003__A021.npz 10

# 无限跑（手动关 viewer）
bash exp_C_passthrough.sh src/assets/motions/g1/amp/WalkandRun/jog_forward_loop_003__A021.npz 0
```

## 文件说明

| 文件 | 用途 |
|---|---|
| `exp_C_passthrough.sh` | 实验 C：纯展示 npz 形态（passthrough 模式）|
| `exp_A_pd.sh` | 实验 A：PD 跟踪 + 真物理仿真 |
| `exp_B_pd_target_zero.sh` | 实验 B：PD 对照组（目标=0） |
| `analyze_symmetry.py` | 左右对称性分析（绘图 + 量化）|
| `AMP_motion_playback.md` | 完整教程文档 |

## 左右对称性分析

```bash
python tutorial/analyze_symmetry.py
```

输出：
- `/tmp/knee_plots/<npz_name>.png`：4 子图（膝/髋 × 原始/对齐）
- 控制台打印对齐前后的相关性和平均误差

**关键指标**：
- 最佳 shift 应 ≈ 半个步态周期（走路是左右交替）
- 对齐后 corr 应 > 0.85（步态形状相似）
- 对齐后 mean\|diff\| 应 0.05~0.25 rad（3°-14°，真实人体差异）

## 对照实验设计

| 实验 | npz 接入？ | 物理？ | 预期 |
|---|---|---|---|
| C. passthrough | ✅ | ❌ | 完美跟随，永不倒 |
| A. pd | ✅ | ✅ | 跟随但 1-2 秒倒 |
| B. pd + target=0 | ❌ | ✅ | 不动，保持 zero pose |

**A vs B 对比**可以严格证明 npz 数据是否真的被物理仿真使用。

## 推荐参数

```bash
# 看清楚数据集本身的样子
mjpython scripts/play_amp_motion_physics.py \
    --npz src/assets/motions/g1/amp/WalkandRun/walk_forward_loop_002__A022.npz \
    --mode passthrough --sim-speed 0.3

# PD 跟随 + 低重力 + 慢放
mjpython scripts/play_amp_motion_physics.py \
    --npz src/assets/motions/g1/amp/WalkandRun/walk_forward_loop_002__A022.npz \
    --mode pd --pd-kp 100 --pd-kd 10 \
    --gravity-scale 0.3 --sim-speed 0.2

# 对照组（不接 npz）
mjpython scripts/play_amp_motion_physics.py \
    --npz src/assets/motions/g1/amp/WalkandRun/walk_forward_loop_002__A022.npz \
    --mode pd --pd-kp 100 --pd-kd 10 \
    --gravity-scale 0.3 --sim-speed 0.2 --target-zero
```

## macOS 注意

必须用 `mjpython` 而不是 `python` 启动 viewer（脚本里已处理）。

## 详细文档

见 `AMP_motion_playback.md`。
