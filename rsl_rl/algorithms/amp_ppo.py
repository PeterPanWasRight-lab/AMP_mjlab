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

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import string_to_callable


class AMPPPO:
    """Proximal Policy Optimization 算法（https://arxiv.org/abs/1707.06347）。

    在标准 PPO 基础上集成了 Adversarial Motion Priors (AMP) 风格奖励塑形。
    通过判别器提供"风格奖励"拉近策略与专家行为。

    Attributes:
        policy: actor-critic 模块，待优化参数。
        discriminator: AMP 判别器，对 (s, s') 转移打分。
        amp_data: 专家 (s, s') 转移数据源（正样本）。
        amp_normalizer: AMP 状态的均值/方差归一化器。
        amp_storage: 策略最近 (s, s') 的回放池（负样本）。
        storage: PPO rollout 存储，存 (obs, action, reward, done, value, log_prob)，
            由 ``act`` / ``process_env_step`` 写入。
    """

    policy: ActorCritic
    """actor-critic 模块。"""

    def __init__(
        self,
        policy,             # ActorCritic 实例，包含 actor / critic 两个 MLP
        discriminator,      # Discriminator 实例，包含 trunk / amp_linear
        amp_data,           # 专家数据源（ReplayBuffer / MotionLoader）
        amp_normalizer,     # AMP 状态归一化器（可 None）
        amp_replay_buffer_size=100000,
        min_std=None,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        optimizer: str = "adam",
        # RND 参数
        rnd_cfg: dict | None = None,
        # Symmetry 参数
        symmetry_cfg: dict | None = None,
        # 多 GPU 训练参数
        multi_gpu_cfg: dict | None = None,
        share_cnn_encoders=False,
    ):
        """初始化 AMP-PPO 算法。

        完成以下工作：
        1. 设置 device（CPU / GPU）和多 GPU 训练配置。
        2. 初始化可选模块：RND（探索增强）、Symmetry（对称性增强）。
        3. 初始化 AMP 组件：判别器、专家数据、策略 replay buffer、归一化器。
        4. 构建优化器（同时优化 policy + discriminator.trunk + discriminator.amp_linear，
           对后两者应用不同 L2 正则）。
        5. 保存 PPO 超参数（clip、gamma、lambda 等）。

        Args:
            policy: actor-critic 模块（如 ActorCritic）。其参数（actor MLP + critic MLP）
                由 PPO 优化。
            discriminator: AMP 判别器。包含 trunk（特征提取 MLP）和 amp_linear
                （最终分类头）。
            amp_data: 专家转移数据源。必须提供
                ``feed_forward_generator(num_mini_batches, mini_batch_size)``，
                输出形状 ``(N, amp_dim)``，其中 ``amp_dim = 2 * num_amp_obs``。
            amp_normalizer: 判别器前向之前对 AMP 状态做归一化，可为 ``None``。
            amp_replay_buffer_size: ``amp_storage`` 的最大容量（FIFO），默认 100,000。
            min_std: 更新后对策略 std 的下界裁剪（数值安全），可为 None。
            num_learning_epochs: 每次 ``update()`` 内的训练 epoch 数。
            num_mini_batches: 每个 epoch 把 rollout 切成多少个 mini-batch。
            clip_param: PPO 截断系数（surrogate objective 的 epsilon）。
            gamma: 折扣因子。
            lam: GAE 的 lambda。
            value_loss_coef: value loss 权重。
            entropy_coef: 策略熵正则权重。
            learning_rate: Adam 优化器学习率。
            max_grad_norm: PPO 梯度的最大范数（裁剪）。
            use_clipped_value_loss: True 用截断版 value loss，否则用 MSE。
            schedule: ``"fixed"``（固定 lr）或 ``"adaptive"``（基于 KL 自适应 lr）。
            desired_kl: 自适应 lr 调度的目标 KL 散度。
            device: PyTorch device，如 ``"cpu"``、``"cuda:0"``。
            normalize_advantage_per_mini_batch: 是否在每个 mini-batch 内归一化 advantage。
            optimizer: 优化器名，目前仅支持 ``"adam"``。
            rnd_cfg: 可选 RND 配置。若提供，RND 内在奖励会加到外部奖励。
            symmetry_cfg: 可选 Symmetry 配置。若提供，会用镜像数据增强 / 镜像 loss。
            multi_gpu_cfg: 可选多 GPU 配置。若提供，会装分布式 all-reduce 钩子。
            share_cnn_encoders: 预留位，AMPPPO 暂未使用。
        """
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # Discriminator components
        self.amploss_coef = 1.0
        self.min_std = min_std
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        params = [
            {"params": self.policy.parameters(), "name": "policy"},
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        """分配 PPO rollout 存储。

        创建 ``self.storage``（``RolloutStorage``），大小为
        ``num_envs * num_transitions_per_env``，用于存并行收集的转移。
        若启用了 RND，也会预留 RND 状态空间。

        Args:
            training_type: ``"rl"``（PPO / AMP-PPO）或 ``"distillation"``（蒸馏）。
            num_envs: 并行环境数 E。
            num_transitions_per_env: 每个环境每次 rollout 采样的步数 T。
            actor_obs_shape: 单条 actor 观测的形状，如 ``[num_actor_obs]``。
            critic_obs_shape: 单条 critic 观测的形状，如 ``[num_critic_obs]``。
            actions_shape: 单条动作的形状，如 ``[num_actions]``。

        Note:
            ``self.amp_storage`` 在 ``__init__`` 中已经分配（不依赖这些维度）；
            这里只创建 ``self.storage``。
        """
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,   # actor观测维度
            critic_obs_shape,  # critic观测维度
            actions_shape,     # 动作输出维度
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs, amp_obs):
        """用当前策略采样动作并暂存 transition。

        调用 actor 采样动作、critic 估计 value，并暂存后续 ``process_env_step``
        需要的字段。**当前时刻**的 ``amp_obs`` 存到 ``self.amp_transition``，
        下一时刻的 ``amp_obs`` 会在 ``process_env_step`` 里和它配对，形成
        ``(s, s')`` 供 AMP 训练使用。

        Args:
            obs: actor 观测张量，形状 ``(E, num_actor_obs)``，输入 actor MLP。
            critic_obs: critic 观测张量，形状 ``(E, num_critic_obs)``，输入 critic MLP。
            amp_obs: AMP 观测张量，形状 ``(E, num_amp_obs)``。
                与 ``obs`` 描述同一时刻，但用判别器所需的 body-frame 格式。

        Returns:
            采样得到的动作张量，形状 ``(E, num_actions)``，已 detach（不参与梯度计算）。
        """
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        """处理一个环境步的反馈，完成当前 transition。

        完成以下工作：
        1. 把外部奖励（通常已含 style_reward）记到 transition。
        2. 若启用了 RND，把内在好奇心奖励加到外部奖励。
        3. 处理 timeout（环境超时但不算 done），加 bootstrap 价值。
        4. 把 ``(prev_amp_obs, curr_amp_obs)`` 插入 ``amp_storage``（判别器负样本）。
        5. 把 transition 加入 ``self.storage`` 并清空，准备下一个 step。
        6. 重置循环策略的 hidden state（若适用）。

        Args:
            rewards: 环境返回的外部奖励，形状 ``(E, 1)``，通常 task_reward + style_reward。
            dones: 环境返回的 done 标志，形状 ``(E,)``，包括 timeout 和真正终止。
            infos: 环境 info 字典。若启用 RND，需要含 ``"observations"["rnd_state"]``。
            amp_obs: **当前时刻**的 AMP 观测，形状 ``(E, num_amp_obs)``。
                与 ``act()`` 时存的 ``prev_amp_obs`` 配对形成 ``(s, s')``。
        """
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        """用 GAE 计算每个时间步的 return 和 advantage。

        用 critic 估计最后一个时间步的 V(s)，然后反向 GAE 计算所有时间步
        的 advantage 和 return，写入 ``self.storage``。

        Args:
            last_critic_obs: 最后一个时间步的 critic 观测，形状
                ``(E, num_critic_obs)``。对应 rollout 边界处的 V(s_T)。
        """
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        """核心训练循环：同时更新 PPO 策略和 AMP 判别器。

        流程：
        1. 构造三路数据生成器并同步迭代：
           - ``generator``: PPO rollout 的 mini-batch（来自 ``self.storage``）
           - ``amp_policy_generator``: 策略最近 (s, s') 转移（来自 ``self.amp_storage``）
           - ``amp_expert_generator``: 专家 (s, s') 转移（来自 ``self.amp_data``）
        2. 对每个 mini-batch：
           a) 重新用当前 policy 算 log_prob、entropy、value
           b) 算 PPO 三件套：surrogate loss、value loss、entropy
           c) 算 AMP 判别器损失：expert_loss / policy_loss / grad_pen_loss
           d) loss = PPO_loss + AMP_loss，一次 ``backward()`` 同时更新 policy 和判别器
        3. 用当前 batch 的 policy / expert 状态更新 ``amp_normalizer`` 的 running stats
        4. 累积每个 mini-batch 的统计量（用于 TensorBoard 日志）

        维度约定（设 N = mini-batch size, E = num_envs, T = num_transitions_per_env）：
        - ``obs_batch``: ``(N, num_actor_obs)``
        - ``critic_obs_batch``: ``(N, num_critic_obs)``
        - ``actions_batch``: ``(N, num_actions)``
        - ``target_values_batch`` / ``returns_batch`` / ``advantages_batch``: ``(N, 1)``
        - ``sample_amp_policy`` / ``sample_amp_expert``:
          ``((N, num_amp_obs), (N, num_amp_obs))``，对应 ``(s, s')``
        - ``policy_d`` / ``expert_d``: ``(N, 1)``

        Returns:
            loss_dict: dict，包含本轮平均的损失值，用于 TensorBoard 日志：
                - ``value_function``: value loss 均值
                - ``surrogate``: PPO surrogate loss 均值
                - ``entropy``: 策略熵均值
                - ``amp``: AMP LSGAN 损失均值
                - ``amp_grad_pen``: 梯度惩罚均值
                - ``amp_policy_pred``: ``policy_d.mean()``，健康训练应向 +1 翻转
                - ``amp_expert_pred``: ``expert_d.mean()``，应接近 +1
                - ``skipped_non_finite_batches``: 因 NaN/Inf 跳过的 batch 数
                - 可选 ``rnd``、``symmetry``
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        skipped_non_finite_batches = 0
        effective_updates = 0

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        # iterate over batches
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
            ) = sample

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            if not torch.isfinite(returns_batch).all() or not torch.isfinite(value_batch).all():
                skipped_non_finite_batches += 1
                continue
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if not torch.isfinite(value_loss):
                skipped_non_finite_batches += 1
                continue

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            if not torch.isfinite(loss):
                skipped_non_finite_batches += 1
                continue

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Discriminator loss.
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)
            loss += self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Keep policy noise above configured floor to avoid invalid Normal std.
            if self.min_std is not None and hasattr(self.policy, "noise_std_type"):
                with torch.no_grad():
                    min_std = torch.as_tensor(self.min_std, device=self.device, dtype=torch.float32)
                    if min_std.ndim == 0:
                        min_std = min_std.unsqueeze(0)

                    if getattr(self.policy, "noise_std_type") == "scalar" and hasattr(self.policy, "std"):
                        target_std = self.policy.std
                        if min_std.numel() == 1:
                            min_std = min_std.expand_as(target_std)
                        elif min_std.numel() != target_std.numel():
                            fallback = torch.clamp_min(min_std.min(), 1.0e-6)
                            min_std = fallback.expand_as(target_std)
                        target_std.clamp_(min=min_std)
                    elif getattr(self.policy, "noise_std_type") == "log" and hasattr(self.policy, "log_std"):
                        target_log_std = self.policy.log_std
                        if min_std.numel() == 1:
                            min_std = min_std.expand_as(target_log_std)
                        elif min_std.numel() != target_log_std.numel():
                            fallback = torch.clamp_min(min_std.min(), 1.0e-6)
                            min_std = fallback.expand_as(target_log_std)
                        target_log_std.clamp_(min=torch.log(torch.clamp_min(min_std, 1.0e-6)))
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            # Store the losses
            effective_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = max(effective_updates, 1)
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
            "skipped_non_finite_batches": float(skipped_non_finite_batches),
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """把 rank 0 的模型参数广播到所有 GPU（多 GPU 训练用）。

        把当前进程（rank 0）的 policy 和（若有）RND predictor 的 state_dict
        广播给所有 GPU，确保所有 rank 拥有相同参数后再开始训练。
        """
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def get_policy(self):
        """返回策略模块。

        Returns:
            ``self.policy``（ActorCritic 实例），供部署/导出时使用。
        """
        return self.policy

    def save(self) -> dict:
        """序列化算法状态用于 checkpoint（v5 格式）。

        将 actor MLP 权重重命名为 ``"mlp.*"``、critic 同理、std 参数重命名为
        ``"distribution.std_param"``，使 checkpoint 兼容统一格式。
        同时保存判别器、optimizer、AMP 归一化器状态。

        Returns:
            dict，键包括：
                - ``actor_state_dict``: actor MLP 权重
                - ``critic_state_dict``: critic MLP 权重
                - ``optimizer_state_dict``: Adam optimizer 状态
                - ``discriminator_state_dict``: 判别器权重（trunk + amp_linear）
                - ``amp_normalizer``: AMP 归一化器状态
        """
        sd = self.policy.state_dict()
        actor_sd, critic_sd = {}, {}
        for k, v in sd.items():
            if k == "std":
                actor_sd["distribution.std_param"] = v
            elif k.startswith("actor."):
                actor_sd["mlp." + k[len("actor."):]] = v
            elif k.startswith("critic."):
                critic_sd["mlp." + k[len("critic."):]] = v
        result = {
            "actor_state_dict": actor_sd,
            "critic_state_dict": critic_sd,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "amp_normalizer": self.amp_normalizer,
        }
        return result

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """从 checkpoint 字典加载算法状态（v5 格式）。

        支持选择性地加载 actor / critic，例如只想部署 actor 可以只传
        ``load_cfg={"actor": True, "critic": False}``。判别器和 AMP 归一化器
        若 checkpoint 中存在则一并加载。

        Args:
            loaded_dict: 由 ``save()`` 产生的 checkpoint 字典。
            load_cfg: 可选，键 ``"actor"``（默认 True）和 ``"critic"``
                （默认与 ``actor`` 一致）控制是否加载对应部分。
            strict: 是否严格匹配 state_dict 的 key。

        Returns:
            True 表示 actor 和 critic 都成功加载（视为恢复训练），否则 False。
        """
        load_cfg = load_cfg or {}
        load_actor = load_cfg.get("actor", True)
        load_critic = load_cfg.get("critic", load_actor)

        sd = self.policy.state_dict()

        if load_actor and "actor_state_dict" in loaded_dict:
            actor_sd = loaded_dict["actor_state_dict"]
            for k, v in actor_sd.items():
                if k == "distribution.std_param" and "std" in sd:
                    sd["std"] = v
                elif k.startswith("mlp."):
                    mapped = "actor." + k[len("mlp."):]
                    if mapped in sd:
                        sd[mapped] = v
                elif k.startswith("distribution.log_std_param") and "std" in sd:
                    sd["std"] = v.exp()

        if load_critic and "critic_state_dict" in loaded_dict:
            critic_sd = loaded_dict["critic_state_dict"]
            for k, v in critic_sd.items():
                if k.startswith("mlp."):
                    mapped = "critic." + k[len("mlp."):]
                    if mapped in sd:
                        sd[mapped] = v

        self.policy.load_state_dict(sd, strict=strict)

        # Load discriminator and AMP normalizer if present
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        if "amp_normalizer" in loaded_dict:
            self.amp_normalizer = loaded_dict["amp_normalizer"]

        return load_actor and load_critic

    def reduce_parameters(self):
        """收集所有 GPU 的梯度并取平均（多 GPU 训练用）。

        在 ``loss.backward()`` 之后调用，把所有 rank 的 policy（和 RND）
        梯度 ``all_reduce`` 求平均，再写回各 rank 的 ``param.grad``，
        保证后续 ``optimizer.step()`` 在每个 rank 上用一致梯度更新。
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
