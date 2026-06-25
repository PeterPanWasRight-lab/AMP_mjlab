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

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCritic(nn.Module):
    """PPO 的 Actor-Critic 网络。

    这是 PPO 算法最核心的策略-价值网络。包含两个独立 MLP：
        - actor (策略网络):  根据本体感知观测 obs，输出动作分布
        - critic (价值网络):  根据特权观测 critic_obs，估计状态价值

    网络结构（默认配置）:
        actor:  Linear(48→256) → ELU → Linear(256→256) → ELU → Linear(256→256) → ELU → Linear(256→29)
        critic: Linear(100→256) → ELU → Linear(256→256) → ELU → Linear(256→256) → ELU → Linear(256→1)

    维度约定（设 E = num_envs 并行环境数）：
        - actor 输入:  (E, num_actor_obs)   ← 本体感知 (关节角、IMU 等)
        - actor 输出:  (E, num_actions)     ← 动作均值
        - critic 输入: (E, num_critic_obs)  ← 特权信息
        - critic 输出: (E, 1)               ← 状态价值 V(s)

    动作采用高斯分布：mean = actor(obs)，std = 可学习参数。
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,           # actor 输入维度（如 48）
        num_critic_obs,          # critic 输入维度（如 100）
        num_actions,             # 动作维度（如 G1 = 29）
        actor_hidden_dims=[256, 256, 256],  # actor 隐层
        critic_hidden_dims=[256, 256, 256], # critic 隐层
        activation="elu",        # 激活函数
        init_noise_std=1.0,      # 初始动作标准差
        noise_std_type: str = "scalar",     # "scalar" 或 "log"
        **kwargs,
    ):
        """初始化 Actor-Critic。

        构建两个 MLP：actor（策略网络）和 critic（价值网络）。
        动作分布用 `Normal(mean, std)`，std 是可学习参数。

        Args:
            num_actor_obs: actor 输入维度（部署时能拿到的）。
            num_critic_obs: critic 输入维度（训练时可用特权信息）。
            num_actions: 动作维度。
            actor_hidden_dims: actor 隐层列表，如 [256, 256, 256]。
            critic_hidden_dims: critic 隐层列表。
            activation: 激活函数名，如 "elu"、"relu"。
            init_noise_std: 初始动作标准差。
            noise_std_type: "scalar" 共享一个 std；"log" 维护 log_std 参数。
            **kwargs: 忽略额外参数。
                def fun(**kwargs) = ** 紧跟参数名 = 打包
                fun(**dict_var) = ** 紧跟变量 = 解包
        """
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # ---------- 构建 actor (策略网络) ----------
        # 动态构建: Linear → activation → Linear → activation → ... → Linear(num_actions)
        # 注意：最后一层不加 activation（动作均值可以是任意实数）
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                # 最后一层: 输出 num_actions 个动作均值
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                # 中间层
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # ---------- 构建 critic (价值网络) ----------
        # 结构与 actor 相同, 但输出维度 = 1 (V(s))
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                # 最后一层: 输出 1 个 V(s)
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # ---------- 动作噪声（标准差）----------
        # 可学习的 std 参数, PPO 训练时会调整
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            # 共享 std: 所有动作维度用同一个 σ
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            # log_std 参数化: 数值更稳定
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # 动作分布（延迟到 update_distribution 创建）
        self.distribution = None
        # 关闭 Normal 的参数校验, 加速 forward。 （比如负概率）
        Normal.set_default_validate_args(False)

    @staticmethod
    # 暂未使用
    def init_weights(sequential, scales):
        """正交初始化 sequential 中所有 Linear 层的权重。"""
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """重置策略状态。MLP 策略无状态, 默认 no-op。"""
        pass

    def forward(self):
        """前向传播。子类需重写, 这里强制报错。"""
        raise NotImplementedError

    @property
    def action_mean(self):
        """当前动作分布的均值。形状 (E, num_actions)。"""
        return self.distribution.mean

    @property
    def action_std(self):
        """当前动作分布的标准差。形状 (E, num_actions)。"""
        return self.distribution.stddev

    @property
    def entropy(self):
        """当前动作分布的熵。形状 (E,)。"""
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """根据观测更新动作分布。

        Args:
            observations: actor 观测, 形状 (E, num_actor_obs)。
        """
        # 计算动作均值
        mean = self.actor(observations)            # (E, num_actions)
        # 计算动作标准差
        if self.noise_std_type == "scalar":
            # clamp_min 防止 std = 0 导致分布退化
            std = torch.clamp_min(self.std, 1.0e-6).expand_as(mean)   # (E, num_actions)
        elif self.noise_std_type == "log":
            std = torch.clamp_min(torch.exp(self.log_std), 1.0e-6).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # 构建多元高斯分布（假设各维度独立）
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        """采样一个动作（带探索噪声）。

        Args:
            observations: actor 观测, 形状 (E, num_actor_obs)。

        Returns:
            采样动作, 形状 (E, num_actions)。
        """
        self.update_distribution(observations)
        return self.distribution.sample()   # 这里无需rsample，只有旧参数采样才会用到这个输出，而旧参数又不参与梯度链

    def get_actions_log_prob(self, actions):
        """计算给定动作在当前分布下的对数概率。

        Args:
            actions: 动作, 形状 (E, num_actions)。

        Returns:
            对数概率, 形状 (E,)。每个样本一个标量, 是 num_actions 个维度的对数概率之和（独立高斯）。
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """推理模式输出动作均值（无探索噪声）。

        用于部署和回放。返回 actor 网络直接输出, 不采样。

        Args:
            observations: actor 观测, 形状 (E, num_actor_obs)。

        Returns:
            动作均值, 形状 (E, num_actions)。
        """
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """用 critic 估计状态价值。

        Args:
            critic_observations: critic 观测, 形状 (E, num_critic_obs)。

        Returns:
            价值 V(s), 形状 (E, 1)。
        """
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """加载 Actor-Critic 参数。

        Args:
            state_dict: 状态字典。
            strict: 是否严格匹配 key。

        Returns:
            True 表示这是 PPO 恢复训练, 完整加载。
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
