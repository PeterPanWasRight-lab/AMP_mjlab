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

#  Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """基于经验均值/方差做在线归一化。

    结构约定：
        - 输入 ``x`` 的形状通常是 ``(B, *shape)``。
          ``B`` 是 batch 维，例如并行环境数 E 或 mini-batch 大小 N。
        - ``shape`` 是单个样本的特征结构，例如 ``num_obs``、``(num_obs,)``，
          或更高维的特征形状。
        - 内部 buffer ``_mean`` / ``_var`` / ``_std`` 的形状是
          ``(1, *shape)``。最前面的 1 用来和 ``(B, *shape)`` 自动广播。
        - ``count`` 是已经参与统计的样本总数，标量 long tensor。

    直觉：
        每次看到一批样本，就更新 running mean / var / std；
        forward 输出 ``(x - mean) / (std + eps)``，输出形状与输入 ``x`` 一样。
    """

    def __init__(self, shape, eps=1e-2, until=None):
        """初始化经验归一化模块。

        Args:
            shape (int or tuple of int): 除 batch 维以外的单样本形状，即 ``*shape``。
                例如观测是 ``(B, 235)`` 时，shape 可以是 ``235`` 或 ``(235,)``。
            eps (float): 数值稳定项，防止除以 0 或过小 std。
            until (int or None): 若指定，当累计样本数 ``count`` 达到该值后停止更新统计量。
        """
        super().__init__()
        self.eps = eps
        self.until = until
        # 若 shape=[num_features]，比如 shape=[235]：
        #   torch.zeros(shape)                  -> 形状 (235,)
        #   torch.zeros(shape).unsqueeze(0)     -> 形状 (1, 235)
        # 更一般地：_mean/_var/_std 的维度都是 (1, *shape)。
        # 例子：
        #   x:     (batch_size, 235)      或 (batch_size, channels, height, width)
        #   _mean: (1,          235)      或 (1,          channels, height, width)
        #   _var:  (1,          235)      或 (1,          channels, height, width)
        #   _std:  (1,          235)      或 (1,          channels, height, width)
        # 保留前导 1 是为了和输入 x: (B, *shape) 在 batch 维上自动广播。
        # register_buffer 会让这些张量跟随 module.to(device)，也会进入 state_dict，但不作为可训练参数。
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        # count: 标量，记录已经用于估计 running statistics 的样本总数。
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        """返回不带 batch 广播维的均值，形状 ``shape``。"""
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        """返回不带 batch 广播维的标准差，形状 ``shape``。"""
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """按当前 running mean/std 归一化输入。

        Args:
            x: 输入张量，形状 ``(B, *shape)``。例如 AMP 状态可能是
                ``(N, num_amp_obs)``。

        Returns:
            归一化后的张量，形状仍为 ``(B, *shape)``。
        """

        if self.training:
            # 训练模式下先用当前 batch 更新统计量，再用更新后的统计量归一化。
            self.update(x)
        # _mean/_std: (1, *shape)，x: (B, *shape)，减法和除法会沿 batch 维广播。
        return (x - self._mean) / (self._std + self.eps)

    # @torch.jit.unused 表示：如果这个 module 被 TorchScript/JIT 编译，当前项目中没用到，因为用了.onnx代替
    # update() 不参与脚本化编译；普通 Python/eager 模式下仍然可以正常调用。
    # 这里 update() 是训练时维护 running statistics 的辅助方法，不是推理图的一部分。
    @torch.jit.unused
    def update(self, x):
        """只更新统计量，不返回归一化结果。

        Args:
            x: 当前 batch，形状 ``(B, *shape)``。

        更新后的结构：
            - ``count``: 旧 count + B
            - ``mean_x`` / ``var_x``: 当前 batch 在 batch 维上的统计，形状 ``(1, *shape)``
            - ``_mean`` / ``_var`` / ``_std``: running statistics，形状 ``(1, *shape)``

        均值/方差融合公式（逐特征维独立计算）：
            设旧统计量为 ``mean_old, var_old, n_old``，
            当前 batch 统计量为 ``mean_batch, var_batch, n_batch``，
            合并后样本数为 ``n_new = n_old + n_batch``，
            ``rate = n_batch / n_new``。

            均值：
                ``mean_new = mean_old + rate * (mean_batch - mean_old)``

            方差（总体方差，unbiased=False）：
                ``var_new = var_old + rate * (var_batch - var_old
                           + (mean_batch - mean_old) * (mean_batch - mean_new))``

            下面代码里的 ``self._mean`` 在更新前是 ``mean_old``，
            更新后是 ``mean_new``，因此方差公式里使用更新后的 ``self._mean``。
        """

        if self.until is not None and self.count >= self.until:
            return

        # count_x: B，本次 batch 的样本数。
        count_x = x.shape[0]
        # self.count 更新前相当于 n_old；更新后相当于 n_new = n_old + count_x。
        self.count += count_x
        # rate: 当前 batch 在“历史样本 + 当前样本”中的占比，是一个标量 tensor。
        # 数学上 rate = n_batch / n_new。
        rate = count_x / self.count

        # var_x/mean_x: (1, *shape)。dim=0 表示只沿 batch 维统计，每个特征维单独维护均值/方差。
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        # delta_mean: (1, *shape)，当前 batch 均值和旧 running mean 的差。
        # 数学上 delta_mean = mean_batch - mean_old。
        delta_mean = mean_x - self._mean
        # 在线更新 running mean；_mean 仍是 (1, *shape)。
        # 对应公式：mean_new = mean_old + rate * delta_mean。
        self._mean += rate * delta_mean
        # 在线更新 running variance。最后一项用于修正“新旧均值移动”带来的方差变化。
        # 对应公式：
        # var_new = var_old + rate * (var_batch - var_old + delta_mean * (mean_batch - mean_new))。
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        # _std: (1, *shape)，供 forward() 中广播除法使用。
        self._std = torch.sqrt(self._var)

    # inverse() 同样是调试/还原尺度用的 Python 辅助方法，不需要进入 TorchScript 推理图。
    @torch.jit.unused
    def inverse(self, y):
        """把归一化后的值还原回原始尺度。

        Args:
            y: 归一化后的张量，形状 ``(B, *shape)`` 或可广播到该结构。

        Returns:
            原始尺度张量，形状与 ``y`` 一致。
        """
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """基于折扣累计回报方差的 reward 归一化。

    来自 Pathak 等人关于大规模 PPO 的经验做法。

    结构约定：
        - 输入 ``rew`` 通常是 ``(E,)`` 或 ``(E, 1)``，E 是并行环境数。
        - ``disc_avg.avg`` 保存每个环境当前的折扣累计奖励，形状与 ``rew`` 一致。
        - ``emp_norm`` 不直接统计原始即时奖励，而是统计折扣累计奖励
          ``avg_t = gamma * avg_{t-1} + rew_t`` 的 running std。

    这样做的目的：
        reward 分布可能不断漂移；用累计回报尺度的 std 来缩放即时奖励，
        可以让 value function 看到更稳定的数值范围。
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        # emp_norm 维护折扣累计奖励的 running std，buffer 形状为 (1, *shape)。
        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        # disc_avg 维护 avg_t = gamma * avg_{t-1} + rew_t，形状跟 rew 一样。
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        """归一化 reward 的尺度。

        Args:
            rew: 当前一步即时奖励，形状通常为 ``(E,)`` 或 ``(E, 1)``。

        Returns:
            缩放后的 reward，形状与 ``rew`` 一致。
        """
        if self.training:
            # update discounected rewards
            # avg: 折扣累计奖励，形状与 rew 一致。
            avg = self.disc_avg.update(rew)

            # update moments from discounted rewards
            # 用 avg 的分布更新 running std，而不是用 rew 本身。
            self.emp_norm.update(avg)

        # emp_norm._std: (1, *shape)，rew 会按 batch/env 维广播相除。
        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class DiscountedAverage:
    r"""折扣累计值维护器。

    这里名字叫 average，但公式本质是折扣累计和：

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    结构约定：
        - ``rew``: 当前步奖励，形状通常为 ``(E,)`` 或 ``(E, 1)``。
        - ``avg``: 与 ``rew`` 同形状，每个环境独立维护一条累计奖励轨迹。

    Args:
        gamma (float): 折扣因子。
    """

    def __init__(self, gamma):
        # 第一次 update 前没有历史累计值；第一次调用时直接用 rew 初始化。
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """用当前奖励更新折扣累计值。

        Args:
            rew: 当前步 reward，形状通常为 ``(E,)`` 或 ``(E, 1)``。

        Returns:
            更新后的 ``avg``，形状与 ``rew`` 一致。
        """
        if self.avg is None:
            self.avg = rew
        else:
            # 对每个环境逐元素更新：avg[i] = gamma * avg[i] + rew[i]。
            self.avg = self.avg * self.gamma + rew
        return self.avg
