# AMP 专家数据集播放与物理仿真 —— 对照实验教程

> **背景**：本教程展示如何用三个对照实验理解 AMP (Adversarial Motion Priors) 中"专家数据集"如何驱动 G1 机器人仿真，并通过三组实验**严格证明 npz 数据确实被物理仿真读取**。

---

## 目录

- [1. 专家数据是什么](#1-专家数据是什么)
- [2. 数据怎么加载到 G1 模型](#2-数据怎么加载到-g1-模型)
- [3. 三个对照实验](#3-三个对照实验)
- [4. 如何运行](#4-如何运行)
- [5. 关键 bug 修复记录](#5-关键-bug-修复记录)
- [6. 核心代码解读](#6-核心代码解读)
- [7. 为什么 AMP 训练后能走](#7-为什么-amp-训练后能走)
- [8. 左右对称性分析](#8-左右对称性分析)
- [9. 附：性能调优建议](#9-附性能调优建议)

---

## 1. 专家数据是什么

### 1.1 数据格式

| 字段 | shape | 含义 |
|---|---|---|
| `fps` | scalar | 采样频率 (50 Hz) |
| `joint_pos` | `(T, 29)` | 每帧 29 个关节的目标角度 (G1 29dof) |
| `body_pos_w` | `(T, 30, 3)` | 每帧 30 个 body 的世界位置 |
| `body_quat_w` | `(T, 30, 4)` | 每帧 30 个 body 的世界姿态 (wxyz) |
| `body_lin_vel_w` | `(T, 30, 3)` | body 世界线速度 |
| `body_ang_vel_w` | `(T, 30, 3)` | body 世界角速度 |

**列数 36 = 3 (base pos) + 4 (base quat) + 29 (joint pos)**，保存在 CSV 阶段。

### 1.2 数据来源

```
[人类动作] LAFAN1 数据集 (7 个受试者, 15 种动作)
    │
    │  IK 运动学重定向
    │  (人体 22 关节 → G1 29 关节)
    ▼
[机器人关节角] 36 列 CSV (base pos + quat + 29 dof)
    │
    │  csv_to_npz.py
    │  用 G1 MuJoCo 模型前向运动学
    ▼
[完整运动学] .npz 文件 (joint_pos + 30 body 状态)
```

**LAFAN1**: 公开的人体动作捕捉数据集 (Baylor 等大学)，30-120 Hz 采样。
**文件命名约定**: `walk_forward_loop_002__A022.npz` 中 `A022` 就是 LAFAN1 受试者编号。

### 1.3 数据集列表

| 文件 | 类型 |
|---|---|
| `walk_forward_loop_002__A022.npz` | 向前走 |
| `jog_forward_loop_003__A021.npz` | 向前慢跑 |
| `walk_backward_loop_001__A022.npz` | 向后走 |
| `walk_sideway_left_loop_002__A021.npz` | 侧向走 |
| `walk_arc_cw_loop_002__A046.npz` | 转弯走 |
| `arc_jog_left_loop_002__A029.npz` | 转弯慢跑 |
| `jog_backward_loop_002__A022.npz` | 后退慢跑 |
| `idle_turn_360_001__A047.npz` | 原地转身 |
| `Recovery/*.npz` | 跌倒恢复 |

---

## 2. 数据怎么加载到 G1 模型

### 2.1 XML 文件选择

| XML | joint 数 | actuator 数 | 用途 |
|---|---|---|---|
| `g1_23dof.xml` | 24 (含 free) | 0 | 仅 23 自由度，缺 6 个 wrist 关节 |
| `g1.xml` | 30 (含 free) | 0 | 完整 29 dof，**只有关节没 actuator** |
| `scene_g1.xml` | 30 (含 free) | **29** | **完整 29 dof + 29 个 motor**（推荐） |

**必须用 `scene_g1.xml` 做真物理实验**。`g1.xml` 没有 actuator，PD 控制无效。

### 2.2 G1 关节顺序（与 npz 的 29 维对应）

```python
G1_29DOF_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
```

### 2.3 关节 vs Actuator 名字差异（容易踩坑）

| 类型 | 命名约定 | 示例 |
|---|---|---|
| joint | `<joint_name>_joint` | `left_hip_pitch_joint` |
| actuator | `<joint_name>` (去 `_joint` 后缀) | `left_hip_pitch` |

**这是 actuator 查找失败 bug 的根源**（见第 5 节）。

---

## 3. 三个对照实验

### 3.1 实验 A：PD + 真物理（接 npz）

**目的**：用 PD 控制让关节跟随 npz，开启真物理仿真。

```
npz[joint_pos][frame]  ──目标──>  PD 控制器  ──力矩──>  actuator
                                                              │
                                                              ▼
                                              mj_step 物理积分 (重力+接触+惯性)
                                                              │
                                                              ▼
                                                        data.qpos 更新
                                                              │
                                                              ▼
                                                         渲染显示
```

**预期行为**：
- ✅ 关节会跟 npz 摆动
- ✅ 力矩不为零（打印显示 50+ Nm）
- ❌ base 没人控制，1-2 秒会倒
- 📉 倒得快慢取决于重力

### 3.2 实验 B：PD + 真物理 + 目标=0（对照组）

**目的**：证明 npz 数据确实在被使用。

唯一区别：PD 目标被强制设为 0，npz 数据**完全不进入控制回路**。

```
PD 控制器目标 = 0  ──>  关节被拉向 zero pose
magnet (不读 npz)
```

**预期行为**：
- ❌ 关节不跟 npz 动
- ❌ 保持 zero pose
- 📉 同样会倒（但没有走路动作）

**如果 A 和 B 行为相同 → 证明 npz 没接上**。
**如果 A 在挥手动脚而 B 不动 → 证明 npz 在驱动**。

### 3.3 实验 C：passthrough（纯展示 npz 形态）

**目的**：完全禁用物理，只展示 npz 数据的真实形态。

```
npz[joint_pos][frame]  ──直接写──>  data.qpos  ──mj_forward 算 FK──>  渲染
(不调 mj_step, 重力=0, 无接触)
```

**预期行为**：
- ✅ 完美跟随 npz
- ✅ 永不倒下
- ✅ 视觉流畅，像看视频
- ⭐ 适合"看数据本身长什么样"

---

## 4. 如何运行

### 4.1 准备

```bash
# macOS 用户特别注意：
# 必须用 mjpython 而不是 python，否则无法打开 viewer
which mjpython  # 应在 mujoco 的环境里
# Linux/Windows 用 python 即可
```

### 4.2 一键运行三个对照实验

```bash
# 实验 A: PD + 真物理（默认跑 30 秒, 自动循环）
bash tutorial/exp_A_pd.sh

# 实验 B: 对照组（不接 npz, 默认 30 秒）
bash tutorial/exp_B_pd_target_zero.sh

# 实验 C: 纯展示 npz（默认 60 秒, 慢速循环 6 遍）
bash tutorial/exp_C_passthrough.sh

# 无限循环（手动关 viewer）
bash tutorial/exp_C_passthrough.sh src/assets/motions/g1/amp/WalkandRun/jog_forward_loop_003__A021.npz 0
```

每个脚本接受两个参数：
- **第 1 个**：npz 文件路径
- **第 2 个**：运行时长（秒），传 0 表示无限

```bash
# 跑 10 秒看某个数据集
bash tutorial/exp_C_passthrough.sh src/assets/motions/g1/amp/WalkandRun/arc_jog_left_loop_002__A029.npz 10

# 无限跑（手动关）
bash tutorial/exp_C_passthrough.sh src/assets/motions/g1/amp/WalkandRun/arc_jog_left_loop_002__A029.npz 0
```

```bash
bash tutorial/exp_C_passthrough.sh src/assets/motions/g1/amp/WalkandRun/jog_forward_loop_003__A021.npz
```

### 4.3 关键参数说明

| 参数 | 默认 | 作用 |
|---|---|---|
| `--mode` | pd | kinematic / passthrough / pd / pd_base |
| `--pd-kp` | 200 | PD 刚度，越大越硬 |
| `--pd-kd` | 8 | PD 阻尼，越大越稳 |
| `--gravity-scale` | 1.0 | 重力缩放，0.3 = 月球重力 |
| `--sim-speed` | 1.0 | 仿真速度，0.2 = 5 倍慢放 |
| `--target-zero` | False | PD 目标恒为 0（实验 B） |
| `--loop` | False | 循环播放 |
| `--duration` | -1.0 | 限时（秒），<=0 表示无限（手动关 viewer） |

### 4.4 推荐参数组合

| 实验目的 | mode | gravity-scale | sim-speed | pd-kp/kd |
|---|---|---|---|---|
| 看脚部接触瞬间 | pd | 1.0 | 0.1 | 100/10 |
| 月球走路体验 | passthrough | 0.16 | 1.0 | - |
| 看平衡崩塌过程 | pd | 0.3 | 0.2 | 100/10 |
| 失重漂浮 | passthrough | 0.05 | 0.3 | - |
| 看 npz 数据本身 | passthrough | 0.0 | 0.3 | - |

### 4.5 一键看完所有数据集

```bash
for f in src/assets/motions/g1/amp/WalkandRun/*.npz; do
    echo "=== $(basename $f) ==="
    mjpython scripts/play_amp_motion_physics.py --npz "$f" --mode passthrough --sim-speed 0.5
done
```

---

## 5. 关键 bug 修复记录

### 5.1 Bug：PD 跟踪误差显示很大，但关节没动

**症状**：
- 实验 A 中视觉上 G1 关节"基本没动"
- 打印 `joint_err(avg)=0.282 rad` 看起来像在跟
- 但倒下时间反常地短（18 帧 ≈ 0.36 秒）

**根因**：
```python
# ❌ 错误写法：actuator 名字 = joint 名字
aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)  
# jn = "left_hip_pitch_joint"
# 但 XML 里 actuator 名字是 "left_hip_pitch"（去掉 _joint 后缀）
# 查找返回 -1，循环里跳过所有 actuator
```

**修复**：
```python
# ✅ 正确写法：去掉 _joint 后缀
actuator_name = jn.replace("_joint", "")
aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
```

**验证**：
- 修复前：`actuator NOT FOUND` × 29，力矩全 0
- 修复后：`actuator_id=0,1,2,3,...`，力矩 50+ Nm

### 5.2 经验教训

1. **不要相信"在动"就以为是"在跟"**：`actual` 变化可能是重力在动，不是 PD 在跟
2. **真证据是力矩**：`data.ctrl[actuator_id]` 不为 0 才算 PD 在工作
3. **mujoco 命名规则**：joint 和 actuator 名字有约定差异，文档里不显眼
4. **怀疑是对的**：用户肉眼发现"关节没动"，比日志数字更可靠

---

## 6. 核心代码解读

### 6.1 PD 模式核心代码

```python
# scripts/play_amp_motion_physics.py 中的 pd 模式

for k, addr in enumerate(joint_qpos_addr):
    if addr < 0 or k >= len(jp) or joint_actuator_addr[k] < 0:
        continue
    # 目标：当前帧 npz 的关节角
    target = float(jp[k])
    # PD 误差
    q_err = target - data.qpos[addr]
    q_vel = data.qvel[joint_qvel_addr[k]]
    # PD 力矩
    torque = pd_kp * q_err - pd_kd * q_vel
    # 写入 actuator 控制信号
    data.ctrl[joint_actuator_addr[k]] = torque

# 物理仿真推进一步
mujoco.mj_step(model, data)
```

### 6.2 Passthrough 模式核心代码

```python
# 关闭重力
model.opt.gravity[2] = 0.0

# 强行覆盖 qpos
for k, addr in enumerate(joint_qpos_addr):
    data.qpos[addr] = float(jp[k])
data.qpos[0:3] = target_base_pos
data.qpos[3:7] = target_base_quat
data.qvel[:] = 0.0

# 只算 forward kinematics，不积分
mujoco.mj_forward(model, data)
```

### 6.3 数据流图

```
                  ┌─────────────────────────────────────┐
                  │  npz: joint_pos[frame, 29]         │
                  │  + body_pos_w[frame, 30, 3]        │
                  │  + body_quat_w[frame, 30, 4]       │
                  └─────────────┬───────────────────────┘
                                │ jp = motion["joint_pos"][frame]
                                ▼
                ┌──────────────────────────────────────┐
                │  PD 控制器 (实验 A/B)                │
                │  target = jp[k]                      │
                │  q_err = target - data.qpos[addr]    │
                │  torque = kp*q_err - kd*q_vel        │
                │  data.ctrl[actuator_id] = torque     │
                └─────────────┬────────────────────────┘
                                │ mj_step
                                ▼
                ┌──────────────────────────────────────┐
                │  MuJoCo 物理引擎                      │
                │  - 重力                              │
                │  - 接触力                            │
                │  - 积分 (qpos += qvel*dt)            │
                └─────────────┬────────────────────────┘
                                │ data.qpos 更新
                                ▼
                ┌──────────────────────────────────────┐
                │  渲染                                │
                │  viewer.sync()                       │
                └──────────────────────────────────────┘
```

---

## 7. 为什么 AMP 训练后能走

### 7.1 实验 A 失败原因分析

| 问题 | 物理原因 |
|---|---|
| base 漂移 | PD 只控制关节，base (floating joint) 没有执行器 |
| 接触力冲击 | 关节被拉到位时速度不为 0，撞地 |
| 累积误差 | 一步失稳 → 下一部 PD 补偿更猛 → 越来越糟 |
| 走路 = 全身协调 | 关节角序列是结果，不是原因 |

### 7.2 AMP 训练做什么

AMP 让策略**学会同时控制全身**：
- actor 网络输出 29 维动作（关节目标角）
- 通过 PD 转换为力矩
- 物理仿真跑一步
- 判别器评估全身运动是否像专家
- PPO 用 style reward + task reward 更新策略

### 7.3 关键 insight

| 强行跟随 (实验 A) | AMP 训练后 |
|---|---|
| 关节角被强制等于 npz | 关节角是**策略**自己决定的 |
| 策略没有任何"补偿"能力 | 策略学会了**主动调整**关节以保持平衡 |
| base 自由，扛不住力矩 | 策略知道 base 会动，会主动调整步态配合 |
| 单方向误差累积 | 闭环反馈，误差可以被修正 |
| 走 0.36 秒就倒 | 能走几分钟不倒 |

**简单说**：强行跟随相当于"把机器人绑在木偶线上，操控它的四肢"——但木偶没有反射和调整能力。AMP 训练相当于"让机器人看视频学舞蹈 + 老师手把手教"——它学会了"怎么动才不会倒"。

---

## 8. 左右对称性分析

> **目的**：验证 AMP 专家数据是否符合真实人体步态的左右对称性。

### 8.1 跑分析

```bash
python tutorial/analyze_symmetry.py
```

输出：
- `/tmp/knee_plots/<npz_name>.png` — 4 子图（膝/髋 × 原始/对齐）
- 控制台打印对齐前后的相关性和平均误差

### 8.2 关键观察

**原始数据：左右腿反相**
- 左膝和右膝 corr 接近 0 或 -1（反相）
- 走路时左右交替是必然的，所以相关性极低

**对齐后：左右腿高度相似**
- 最佳 shift 接近**半个步态周期**（步态左右交替的物理结果）
- 对齐后 corr 跳到 0.87~0.98
- 对齐后 mean|diff| 在 0.05~0.22 rad（3°-13°）

| 数据集 | 膝关节 best shift | 对齐后 corr | 对齐后 mean\|diff\| |
|---|---|---|---|
| walk_forward | +620ms (半周期) | +0.872 | 0.120 rad (6.9°) |
| jog_forward | +400ms (半周期) | +0.884 | 0.169 rad (9.7°) |
| arc_walk_left | -540ms (半周期) | +0.979 | 0.074 rad (4.2°) |

### 8.3 真实人体动捕的强证据

| 数据类型 | 对齐后 mean\|diff\| | 解读 |
|---|---|---|
| 合成完美对称数据 | < 0.01 rad | 失真，不真实 |
| 动画/插值数据 | > 0.5 rad (30°+) | 太不规则 |
| **真实人体数据** | **0.05~0.25 rad (3°-14°)** | **匹配本数据集** |

**3°-14° 的微差**正是真实人体步态的特征：左右腿活动范围相近但有微调（优势侧/非优势侧）。强制对称化（diff=0）和明显不对称（diff>20°）都不符合真实情况。

---

## 9. 附：性能调优建议

### 8.1 PD 参数调优

| kp | kd | 适用场景 |
|---|---|---|
| 50 | 5 | 软跟随，能看到关节响应延迟 |
| 100 | 10 | 平衡点（推荐） |
| 200 | 15 | 硬跟随 |
| 500 | 30 | 几乎锁定关节 |
| 2000 | 80 | 接触力大、容易抖振 |

### 8.2 重力与速度的组合

| 目的 | gravity-scale | sim-speed | 视觉效果 |
|---|---|---|---|
| 看清楚数据集 | 0.0 | 0.3 | 漂浮的"电影" |
| 真实演示 | 1.0 | 1.0 | 1 秒倒 |
| 月球 | 0.16 | 1.0 | 6 倍弹跳 |
| 慢动作看接触 | 1.0 | 0.1 | 10 倍慢放 |
| 平衡分析 | 0.3 | 0.2 | 慢放 5 倍 + 月球 |

### 8.3 常见错误排查

| 现象 | 原因 | 修复 |
|---|---|---|
| Viewer 打不开 | macOS 用 python 而非 mjpython | `mjpython` 启动 |
| 关节不动 | actuator 名字错误 | 用 `joint_name.replace("_joint", "")` |
| 一下就倒 | base 自由 + 重力 | 加 `--gravity-scale 0.3` |
| 关节抖振 | kp 太大 | 减小 `--pd-kp` |
| 关节跟不上 | kp 太小 | 增大 `--pd-kp` |

---

## 9. 引用与参考

- **AMP 论文**: Peng et al., "Adversarial Motion Priors for Stylized Physics-Based Character Control", SIGGRAPH 2021
- **LAFAN1 数据集**: https://huggingface.co/datasets/lmbspecial/lafan1
- **G1 机器人**: Unitree G1 (29 dof)
- **MuJoCo 文档**: https://mujoco.readthedocs.io/
- **mjlab 框架**: 本项目 `/Users/peterpan/PeterPanWorkspace/AMP_mjlab/`

---

**最后更新**: 2026-06-18
**作者**: Peter Pan (with CodeBuddy AI assistance)
