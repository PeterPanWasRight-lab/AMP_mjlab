# D 太强 → G 学不到的严格数学推导

## 1. 风格奖励函数

AMP 判别器 $D$ 的输出 $d$ 通过 LSGAN 训练后，在 $[-1, +1]$ 范围内。
推理时由 $d$ 计算风格奖励：

$$r_{\text{style}}(s, s') = \max\left(0, \; 1 - \frac{1}{4}\big(D(s, s') - 1\big)^2\right)$$

为简化分析，略去 clip 操作（当 $D \ge -1$ 时不会触发），即：

$$r_{\text{style}}(s, s') = 1 - \frac{1}{4}\big(D(s, s') - 1\big)^2$$

---

## 2. 判别器的 LSGAN 训练目标

判别器 $D_\phi$ 的参数为 $\phi$，其训练目标为：

$$\mathcal{L}_D = \frac12 \mathbb{E}_{(s_e,s_e')\sim\mathcal{B}_e}\left[(D_\phi(s_e,s_e') - 1)^2\right] + \frac12 \mathbb{E}_{(s_\pi,s_\pi')\sim\mathcal{B}_\pi}\left[(D_\phi(s_\pi,s_\pi') + 1)^2\right] + \lambda \mathbb{E}_{(s_e,s_e')}[\|\nabla D_\phi\|^2]$$

训练收敛后，$D$ 会对：
- 专家样本 $\rightarrow$ 输出 $+1$
- 策略样本 $\rightarrow$ 输出 $-1$

---

## 3. "D 太强" 的定义

**D 太强** = D 对**所有**策略样本都精准输出 $-1$：

$$D_\phi(s, s') \approx -1 \quad \forall (s, s') \sim \pi_\theta$$

代入风格奖励：

$$r_{\text{style}} = 1 - \frac14((-1) - 1)^2 = 1 - \frac14 \cdot 4 = 0$$

**结论**：D 太强时，$\boxed{r_{\text{style}} \approx 0}$，对所有策略样本都几乎为 0。

---

## 4. 策略梯度分析

策略 $\pi_\theta$ 通过 PPO 最大化期望总回报：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t \big(r_{\text{task}} + r_{\text{style}}\big)\right]$$

PPO 的策略梯度近似为：

$$\nabla_\theta J(\theta) \approx \mathbb{E}_{s, a \sim \pi_{\theta_{\text{old}}}}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)\right]$$

优势函数 $A(s,a)$ 可拆为 task 和 style 两部分：

$$A(s, a) = A_{\text{task}}(s, a) + A_{\text{style}}(s, a)$$

其中：

$$A_{\text{style}}(s, a) = r_{\text{style}}(s, s') + \gamma V_{\text{style}}(s') - V_{\text{style}}(s)$$

当 D 太强时，对所有 $(s,s')$ 有 $r_{\text{style}} \approx 0$，因此：

$$A_{\text{style}}(s, a) \approx \gamma V_{\text{style}}(s') - V_{\text{style}}(s)$$

由 $r_{\text{style}} \approx 0$，值函数 $V_{\text{style}} \approx 0$，因此：

$$\boxed{A_{\text{style}}(s, a) \approx 0}$$

策略梯度退化为：

$$\nabla_\theta J(\theta) \approx \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A_{\text{task}}(s, a)\right]$$

**风格维度上的梯度完全消失**——G1 只能从 task reward 学习。

> **关键洞察**：不是 $r_{\text{style}} = 0$ 导致不能学习——而是 $\text{Var}[A_{\text{style}}] = 0$ 导致无法学习。PPO 需要**有差异的优势信号**来区分"好动作"和"坏动作"；当所有动作的 $A_{\text{style}}$ 都为 0 时，PPO 不知道哪个动作更像人。

---

## 5. 严格数学推导（$\epsilon$ 形式化）

**定理**：设判别器 $D_\phi$ 对任意 $(s,s') \sim \pi_\theta$ 满足 $|D_\phi(s,s') + 1| < \epsilon$，则
$$|r_{\text{style}}(s,s')| < \epsilon + \frac14 \epsilon^2 = O(\epsilon)$$
且
$$\|\nabla_\theta \mathbb{E}_{\pi_\theta}[r_{\text{style}}]\| = O(\epsilon)$$

当 $\epsilon \to 0$ 时，风格奖励对策略参数的梯度以 $\epsilon$ 速度消失。

**证明**：

**Step 1**：令 $D = -1 + \delta$，其中 $|\delta| < \epsilon$。

代入风格奖励公式：

$$
\begin{aligned}
r_{\text{style}} &= 1 - \frac14(D - 1)^2 \\
&= 1 - \frac14(-1 + \delta - 1)^2 \\
&= 1 - \frac14(\delta - 2)^2 \\
&= 1 - \frac14(\delta^2 - 4\delta + 4) \\
&= 1 - \frac14\delta^2 + \delta - 1 \\
&= \delta - \frac14\delta^2
\end{aligned}
$$

因此 $|r_{\text{style}}| = |\delta - \frac14 \delta^2| \le |\delta| + \frac14|\delta|^2 < \epsilon + \frac14\epsilon^2 = O(\epsilon)$。

**Step 2**：由链式法则：

$$\nabla_\theta \mathbb{E}_{\pi_\theta}[r_{\text{style}}] = \mathbb{E}_{s,s' \sim \pi_\theta}\left[\frac{\partial r_{\text{style}}}{\partial D} \cdot \frac{\partial D}{\partial s} \cdot \frac{\partial s}{\partial \theta} + \frac{\partial r_{\text{style}}}{\partial D} \cdot \frac{\partial D}{\partial s'} \cdot \frac{\partial s'}{\partial \theta}\right]$$

其中：

$$\frac{\partial r_{\text{style}}}{\partial D} = -\frac12(D - 1) = -\frac12(-1 + \delta - 1) = -\frac12(\delta - 2) = 1 - \frac\delta2$$

当 $\delta \approx 0$ 时，$\frac{\partial r_{\text{style}}}{\partial D} \approx 1$。

但关键是 $\frac{\partial D}{\partial s}$ 和 $\frac{\partial D}{\partial s'}$：由于 D 在策略样本附近已训练到饱和（输出永远 $-1$），$\frac{\partial D}{\partial(s,s')} \approx 0$。

因此 $\|\nabla_\theta \mathbb{E}[r_{\text{style}}]\| \approx 1 \cdot 0 \cdot \frac{\partial s}{\partial \theta} = 0$，即 $O(\epsilon)$。$\square$

---

## 6. 与 D 平衡时的对比

| 状态 | D 对策略输出 $d$ | $r_{\text{style}}$ | $A_{\text{style}}$ 方差 | 学习能力 |
|---|---|---|---|---|
| **D 太强** | $\approx -1$ | $\approx 0$ | $\approx 0$ | ❌ 学不到 |
| **D 平衡** (80%) | $\approx 0 \sim +0.3$ | $\approx 0.6 \sim 0.8$ | $> 0$ | ✅ 学到风格 |
| **D 太弱** | $\approx +0.5$ | $\approx 0.9$ | $> 0$ 但无区分度 | ⚠️ 学偏 |

**D 平衡时**，策略样本落在"过渡区"：

$$\mathbb{E}_{(s,s')\sim\pi_\theta}[D(s,s')] \in [0, +0.3]$$

$$\mathbb{E}_{(s,s')\sim\pi_\theta}[r_{\text{style}}] \in [0.6, 0.8]$$

$$\text{Var}[A_{\text{style}}] > 0$$

策略梯度中 $A_{\text{style}}$ 项有**非零方差**，PPO 可以据此调整策略参数：

$$\nabla_\theta J(\theta) = \underbrace{\mathbb{E}[\nabla_\theta \log \pi \cdot A_{\text{task}}]}_{\text{任务信号}} + \underbrace{\mathbb{E}[\nabla_\theta \log \pi \cdot A_{\text{style}}]}_{\text{风格信号}} \neq 0$$

---

## 7. 结论

> **"D 太强 → G 学不到" 的数学证明**：
>
> 当 $D$ 对所有策略样本输出 $-1$ 时，$r_{\text{style}}$ 恒为 0，$A_{\text{style}}$ 的方差为 0，风格维度上的策略梯度 $\|\nabla_\theta \mathbb{E}[r_{\text{style}}]\| = O(\epsilon)$ 消失。
>
> PPO 需要优势函数有非零方差才能学习，当 $A_{\text{style}}$ 处处为 0 时，风格维度上的学习信号完全消失。

---

## 附录：为什么"$\text{Var}=0$"比"$r=0$"更重要

如果 $r_{\text{style}}$ 始终为 0.5 但**恒定不变**（$\text{Var}=0$）：

```
第 1 步: r_style = 0.5, d = 0, A_style = 0.1
第 2 步: r_style = 0.5, d = 0, A_style = 0.1
第 3 步: r_style = 0.5, d = 0, A_style = 0.1
... 所有步都一样

→ 策略: "所有动作得到的风格信号都一样，没区别"
→ 无法学习
```

如果 $r_{\text{style}}$ 在 0.3~0.8 之间变化（$\text{Var} > 0$）：

```
第 1 步: r_style = 0.8, d = +0.3, A_style = +0.2  ← 好动作
第 2 步: r_style = 0.3, d = -0.5, A_style = -0.3  ← 坏动作

→ 策略: "第二步那种动作不好，第一步那种更好"
→ 可以学习!
```

即：PPO 需要区分，不需要绝对值。
