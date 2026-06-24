# AMP 判别器原理可视化

本目录包含**五个**文档，把 AMP 判别器的数学原理、训练动力学、数学框架和代码数据结构展示出来。

## 文件

| 文件 | 类型 | 内容 |
|---|---|---|
| `index.html` | 网页 | 判别器原理总览（9 节）：D 是什么、训练损失、风格奖励、准确率影响、对抗平衡、数据流、公式、概念澄清、总结 |
| `two_questions.html` | 网页 | 两个核心问题解答：① D 太强为何奖励 ≈ 0 ② 为什么不换更好的奖励 |
| `math_framework.html` | 网页 | **新** AMP 数学框架：min-max 博弈、Nash 平衡、收敛的数学定义、四股力量、交互式模拟 |
| `replay_buffer.html` | 网页 | ReplayBuffer 完整可视化：环形缓冲区 insert、feed_forward_generator 采样、代码逐行注释 |
| `d_too_strong_proof.md` | 文档 | D 太强 → G 学不到的严格数学推导（含 $\epsilon$ 形式化证明）|

## 打开方式

```bash
# macOS - 网页
open tutorial/判别器/index.html
open tutorial/判别器/two_questions.html
open tutorial/判别器/math_framework.html
open tutorial/判别器/replay_buffer.html
```

## 五个文档的层次

```
index.html             (入门：判别器是什么)
     ↓
two_questions.html     (问题：D 太强 / 不换更好奖励)
     ↓
math_framework.html    (理论：min-max 博弈 + Nash 平衡)
     ↓
d_too_strong_proof.md  (深入：D 太强梯度消失的数学证明)
     ↓
replay_buffer.html     (代码：ReplayBuffer 数据结构的完整可视化)
```

## math_framework.html 的 8 节内容

1. **AMP 完整数学框架** — min-max 形式
2. **"收敛"的三种定义** — 参数收敛、分布收敛、Nash 平衡收敛
3. **Nash 平衡的精确定义** — 形式化 + 存在性定理
4. **平衡的四个数学条件** — LSGAN/grad_pen/骗 D/做任务
5. **交互式 Nash 平衡模拟** — Plotly 实时曲线
6. **训练健康的诊断指标** — 6 个数值标准
7. **修复方法** — 5 种数学调整
8. **总结** — "G 无法收敛"的数学含义

## 核心交互演示

**第 5 节 Nash 平衡模拟**：
- 3 个滑块：初始 G 能力、初始 D 能力、grad_pen λ
- "开始训练 100 步"按钮：实时更新 G/D 能力曲线、d_policy、reward
- 自动诊断状态：D 支配 / 健康 / G 过强

**实时显示**：
- 第一图：G 能力 vs D 能力曲线
- 第二图：d_policy（黄虚线）vs style_reward（绿实线）
- 状态框：用数学语言描述当前训练状态

## "G 无法收敛" 的数学含义（精要）

$$\min_{\theta_G} \max_{\theta_D} \mathcal{L}(G, D) \quad \text{在 Nash 平衡} (G^*, D^*) \text{处达到稳态}$$

**G 无法收敛** = 策略参数 $\theta_G$ 不趋向 $\theta_G^*$：
- $\nabla_\theta J(\theta) \approx 0$（梯度消失）
- $A(s,a) \approx 0$（优势函数为零）
- $D$ 准确率 > 95%（判别面太尖锐）

**修复** = 把系统拉回 Nash 平衡点：
- 增大 grad_pen λ
- 降低 D 学习率 $\eta_D$
- 减小 D 网络容量
- 降低 G 学习率 $\eta_G$

## 关键诊断数值

| 指标 | 健康值 | 危险值 | 含义 |
|---|---|---|---|
| $d_{\text{expert}}$ | 0.7~0.9 | > 0.95 | $\mathbb{E}[D(s_e, s_e')]$ |
| $d_{\text{policy}}$ | 0.3~0.7 | < -0.5 | $\mathbb{E}[D(s_\pi, s_\pi')]$ |
| $d_{\text{expert}} - d_{\text{policy}}$ | 0.3~0.6 | > 1.0 | D 判别力 |
| $\bar{r}_{\text{style}}$ | 0.5~0.9 | < 0.2 | 平均 style reward |
| $\mathcal{H}(\pi)$ | 缓慢下降 | 不变/上升 | 策略熵 |

## 离线使用

依赖通过 CDN 加载（MathJax、Plotly）。如需离线使用：
1. 下载 MathJax 和 Plotly 到本地
2. 修改 HTML 文件中的 `<script>` 标签路径
