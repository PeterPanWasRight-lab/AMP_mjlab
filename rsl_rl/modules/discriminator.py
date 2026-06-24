# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    """AMP 判别器网络。

    AMP 的核心组件。输入 (s, s') 拼接后的 390 维向量，输出一个标量 d ∈ [-1, +1]，
    表示"这个动作过渡像不像人"（+1 = 像专家，-1 = 像策略）。

    网络结构：
        [输入: 390 维] → trunk(MLP) → amp_linear(Linear, 单输出) → [1 个分数 d]

    它还提供了两个关键功能：
        - compute_grad_pen: 梯度惩罚，防止判别器太强（D 支配）
        - predict_amp_reward: 将 d 转换为风格奖励，注入到环境 reward

    输入维度约定（本项目中）：
        - input_dim = 390 = 2 × 195（s + s' 拼接）
        - 195 维 = 13 个 body × 15 特征（pos/ori/vel 各 3+6+3+3）
    """

    def __init__(
        self,
        input_dim,            # 输入维度（拼接后的，如 390）
        amp_reward_coef,      # 风格奖励缩放系数，控制 style_reward 的权重
        hidden_layer_sizes,   # trunk MLP 隐层大小，如 [1024, 512, 256]
        device,               # torch 设备
        task_reward_lerp=0.0, # 任务奖励混合系数（0=纯风格，1=纯任务）
    ):
        """初始化判别器。

        构建 trunk（MLP 特征提取器）和 amp_linear（分类头）。

        Args:
            input_dim: 输入维度。训练时 = 390（[s, s'] 拼接），推理时 += 390。
            amp_reward_coef: 风格奖励的缩放系数，如 0.5。
                最终 reward = coef × clamp(1 - 0.25(d-1)², 0)。
            hidden_layer_sizes: trunk 的隐层维度列表，如 [1024, 512, 256]。
            device: torch device，如 'cuda:0'。
            task_reward_lerp: task reward 混合比例，0~1。
                0.75 表示 75% 任务奖励 + 25% 风格奖励（推荐值）。
        """
        super().__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef

        # ---------- 构建 trunk MLP ----------
        # trunk 负责"特征提取"：把 390 维输入压缩成 256 维特征
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())          # 激活函数
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)     # MLP 主干

        # ---------- 构建 amp_linear 分类头 ----------
        # 把 256 维特征 → 1 个标量（分数）
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        # 设为训练模式（PyTorch 默认就是 train，此处显式声明）
        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        """判别器前向传播。

        输入 (s, s') 拼接向量，输出一个标量分数 d。

        Args:
            x: 输入张量，形状 (batch_size, input_dim)，即 (N, 390)。

        Returns:
            d: 判别器分数，形状 (batch_size, 1)。
               - 专家样本 → 训练目标是 +1
               - 策略样本 → 训练目标是 -1
        """
        h = self.trunk(x)           # trunk 特征提取: (N, 390) → (N, 256)
        d = self.amp_linear(h)      # 分类头: (N, 256) → (N, 1)
        return d

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        """计算专家数据上的梯度惩罚（防止 D 太强的关键设计）。梯度惩罚的原始动机：让 D 在"有意义的区域"平滑
           策略样本（散布各处）
                ×    ×
           ×         ×
                ×
           专家流形（紧凑聚集）
           ●●●●
           ●●●●     ← 只在这里做梯度惩罚
           ●●●●

        核心思想：
            D 在专家数据附近的输出应该"平滑"——输入微小变化不会导致输出剧烈波动。
            这样 G1 才能收到光滑的风格奖励信号，而不是跳跃的 0/1。

        数学公式：
            L_gp = λ · E[ ||∇_x D(x)||² ],  其中 x = [expert_state, expert_next_state]

        计算过程：
            1. 把 (s, s') 拼接成 x
            2. 打开 x 的梯度追踪
            3. 计算 D(x) 对 x 的梯度
            4. 梯度范数 → 平方 → 平均 → × λ

        Args:
            expert_state: 专家当前状态，形状 (N, num_amp_obs)。
            expert_next_state: 专家下一状态，形状 (N, num_amp_obs)。
            lambda_: 梯度惩罚系数，默认 10。

        Returns:
            标量损失。
        """
        # 拼接成判别器输入: (N, 390)
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True      # 打开梯度追踪，为了算 ||∇D||

        # 判别器前向: (N, 390) → (N, 1)
        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)

        # 计算梯度: ∂D/∂x
        # grad_outputs=[2, 3, 5] 意思是：
        # loss = 2*y[0] + 3*y[1] + 5*y[2]
        # 然后求 ∂loss/∂x
        grad = autograd.grad(
            outputs=disc,        # 对哪个输出求导 → D(x)
            inputs=expert_data,  # 对哪个输入求导 → x
            grad_outputs=ones,   # 反向传播时 loss 的梯度（全 1）。起点: ∂loss/∂loss = 1
            create_graph=True,   # 保留计算图（让 grad_pen 也参与梯度传播）
            retain_graph=True,   # 不释放计算图（上面还有 LSGAN loss 要用）
            only_inputs=True,    # 只对 expert_data 求导，不对其他输入
        )[0]                     # (N, 390)

        # 梯度惩罚: λ * ||∇D(x)||² 的均值
        # 目标是让梯度范数接近 0 → D 在专家附近"平"
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        """用当前判别器预测风格奖励（推理时调用）。

        这个函数在环境 wrapper 的 step() 中被调用，用来把判别器的 d 分数
        转换为风格奖励，加到 task reward 上。

        转换公式：
            style_reward = amp_reward_coef × clamp(1 - 0.25 × (d - 1)², min=0)

        其中 d 是判别器对当前 (s, s') 的打分。

        Args:
            state: 当前状态，形状 (E, num_amp_obs)。
            next_state: 下一状态，形状 (E, num_amp_obs)。
            task_reward: 任务奖励，形状 (E,) 或 (E, 1)。
            normalizer: 可选。如果是 AMP 训练模式，会用 running stats 归一化。

        Returns:
            tuple:
                - reward: 风格奖励，形状 (E,)。d 越接近 +1（像人）→ 奖励越大。
                - d: 判别器原始输出，形状 (E, 1)。可用来监控训练进度。
        """
        with torch.no_grad():   # 推理时不计算梯度（节省显存）
            self.eval()         # 切换到评估模式（不影响 conv/dropout，但语义上明确）

            # 可选：归一化
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            # 判别器打分: (E, 390) → (E, 1)
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))

            # 把 d 转换为风格奖励
            #   d = +1 → reward = coef × 1           （满分像人）
            #   d = 0  → reward = coef × 0.75        （还行）
            #   d = -1 → reward = coef × 0           （完全不像人）
            reward = self.amp_reward_coef * torch.clamp(
                1 - (1 / 4) * torch.square(d - 1), min=0
            )

            # 可选：与任务奖励线性混合
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))

            self.train()        # 切回训练模式（为下一次前向做准备）

            reward = reward.squeeze(-1)  # (E, 1) → (E,)
        return reward, d

    def _lerp_reward(self, disc_r, task_r):
        """线性混合风格奖励和任务奖励。

        公式：
            r = (1 - lerp) × style_reward + lerp × task_reward

        Args:
            disc_r: 风格奖励，形状 (E, 1)。
            task_r: 任务奖励，形状 (E, 1)。

        Returns:
            混合后的奖励，形状 (E, 1)。
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
