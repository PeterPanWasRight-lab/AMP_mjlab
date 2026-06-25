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

from rsl_rl.utils import split_and_pad_trajectories


"""
RolloutStorage 数据结构总览
==========================

这个类保存 on-policy 算法一次 rollout 收集到的数据。可以把它想成一张二维表：

    时间维 T = num_transitions_per_env
    环境维 E = num_envs

    ┌────────────────────────────── 环境维 E ──────────────────────────────┐
    │ env0              env1              env2                    envE-1   │
┌───┼──────────────────────────────────────────────────────────────────────┤
│t0 │ obs[0,0]          obs[0,1]          obs[0,2]        ...     obs[0,E-1]
│t1 │ obs[1,0]          obs[1,1]          obs[1,2]        ...     obs[1,E-1]
│t2 │ obs[2,0]          obs[2,1]          obs[2,2]        ...     obs[2,E-1]
│...│ ...               ...               ...             ...     ...
│T-1│ obs[T-1,0]        obs[T-1,1]        obs[T-1,2]      ...     obs[T-1,E-1]
└───┴──────────────────────────────────────────────────────────────────────┘
 时间维 T

每个字段都按相同的前两维组织：
    observations:             (T, E, *obs_shape)
    privileged_observations:  (T, E, *privileged_obs_shape)
    actions:                  (T, E, *actions_shape)
    rewards:                  (T, E, 1)
    dones:                    (T, E, 1)

PPO 训练额外保存：
    values:                   (T, E, 1)        旧 critic 的 V(s_t)
    actions_log_prob:         (T, E, 1)        旧 policy 下整条动作的 log_prob
    mu / sigma:               (T, E, *A)       旧 policy 的动作高斯均值/标准差
    returns / advantages:     (T, E, 1)        GAE 算出来的训练目标

feed-forward 策略训练时，会把前两维展平：
    (T, E, ...) -> (T * E, ...)
然后随机抽 mini-batch。

recurrent 策略训练时，不能简单打乱每个时间步，因为 RNN 需要连续轨迹；
所以会先按 done 切成 episode 片段，再 pad 成同长度轨迹 batch。
"""


class RolloutStorage:
    class Transition:
        """单步临时容器。

        Runner/Algorithm 每和环境交互一步，会先把这一时刻的数据放进 Transition：
            observations:             (E, *obs_shape)
            privileged_observations:  (E, *privileged_obs_shape)
            actions:                  (E, *actions_shape)
            rewards:                  (E,) 或 (E, 1)
            dones:                    (E,)
            values:                   (E, 1)
            actions_log_prob:         (E,) 或 (E, 1)
            action_mean/action_sigma: (E, *actions_shape)
            rnd_state:                (E, *rnd_state_shape)，可选

        add_transitions() 会把它写入 RolloutStorage 的第 self.step 行：
            storage[field][self.step] <- transition.field
        """

        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None
            # self.gt = None
        def clear(self):
            """清空临时 transition，避免下一步误用旧数据。"""
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        """创建 rollout buffer。

        Args:
            training_type: "rl" 或 "distillation"。
            num_envs: 并行环境数 E。
            num_transitions_per_env: 每个环境收集步数 T。
            obs_shape: 单个 actor 观测形状，例如 [Oa]。
            privileged_obs_shape: 单个 critic/teacher 观测形状，例如 [Oc]。
            actions_shape: 单条动作形状，例如 [A]。
            rnd_state_shape: RND 输入状态形状，例如 [Ornd]，不用 RND 时为 None。
            device: 存储张量所在设备。
        """
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape

        # Core
        # observations[t, e] 是第 e 个环境在 rollout 第 t 步的 actor 观测。
        # 形状：(T, E, *obs_shape)，例如 (24, 4096, 235)。
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape is not None:
            # privileged_observations[t, e] 是 critic/teacher 用的额外观测。
            # 形状：(T, E, *privileged_obs_shape)。
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        # rewards/dones 都统一存成最后一维为 1，便于和 values/returns/advantages 对齐。
        # rewards: (T, E, 1)，dones: (T, E, 1)。
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # actions: (T, E, *actions_shape)，例如动作 29 维时为 (T, E, 29)。
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            # 蒸馏时额外保存 teacher/privileged policy 给出的目标动作。
            # privileged_actions: (T, E, *actions_shape)。
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type == "rl":
            # values: 旧 critic 在 rollout 时估计的 V(s_t)，形状 (T, E, 1)。
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            # actions_log_prob: 旧 policy 下采样动作的联合 log_prob，形状 (T, E, 1)。
            # 注意：如果动作是 A 维，A 维 log_prob 已经 sum 成一个标量。
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            # mu/sigma: 旧 policy 的高斯分布参数，形状 (T, E, *actions_shape)。
            # PPO 自适应 KL 会用它和新 policy 的均值/标准差比较。
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            # returns/advantages: compute_returns() 之后填充，形状 (T, E, 1)。
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RND
        if rnd_state_shape is not None:
            # rnd_state: (T, E, *rnd_state_shape)，用于 RND predictor/target 训练。
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        # For RNN networks
        # recurrent policy 会在每个时间步保存 actor/critic 的 hidden state。
        # 具体形状取决于 GRU/LSTM，一般可理解成：
        # saved_hidden_states_*[layer_or_gate]: (T, num_layers, E, hidden_dim)。
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        # step 是当前写入的时间行，范围 0..T。每调用一次 add_transitions() 加 1。
        self.step = 0

    def add_transitions(self, transition: Transition):
        """把一个环境步的数据写入 buffer 的第 self.step 行。

        transition 中每个字段通常带环境维 E：
            transition.observations:     (E, *obs_shape)
            transition.actions:          (E, *actions_shape)
            transition.rewards/dones:    (E,) 或 (E, 1)

        写入后 storage 中对应字段变成：
            self.observations[self.step]: (E, *obs_shape)
        """
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        # 第 self.step 行对应 rollout 的当前时间步 t。
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        # rewards/dones 可能从环境来时是 (E,)，这里 view(-1, 1) 统一成 (E, 1)。
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            # transition.actions_log_prob 可能是 (E,)，统一存为 (E, 1)。
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RND
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        """保存 RNN hidden state。

        hidden_states 通常是 (actor_hidden, critic_hidden)。
        - GRU: hidden 形状一般是 (num_layers, E, hidden_dim)。
        - LSTM: hidden 是 (h, c) 元组，每个都是 (num_layers, E, hidden_dim)。

        为了统一处理，GRU hidden 会被包装成单元素 tuple。
        最终 saved_hidden_states_a/c 的每个元素形状是：
            (T, num_layers, E, hidden_dim)
        第一维 T 用来记录每个 rollout 时间步进入 RNN 前的 hidden state。
        """
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            # self.observations.shape[0] == T。这里在 hidden state 前面补一个时间维。
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        """开始下一轮 rollout 前重置写入指针。

        注意：这里不清零大张量内容，只把 step 置 0。下一轮 add_transitions()
        会从第 0 行开始覆盖旧数据，避免不必要的显存写入。
        """
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        """用 GAE(lambda) 计算 returns 和 advantages。

        Args:
            last_values: rollout 结束后 critic 对最后状态的估计，形状 (E, 1)。
            gamma: 折扣因子。
            lam: GAE 的 lambda。
            normalize_advantage: 是否对整个 (T, E, 1) advantage 做标准化。

        相关张量：
            rewards[step]: (E, 1)
            values[step]:  (E, 1)
            dones[step]:   (E, 1)
            advantage:     (E, 1)，反向递推中保存下一步累计 advantage。
        """
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            # delta: (E, 1)，每个环境当前时间步一个 TD 误差。
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            # 如果当前环境 done，则 next_is_not_terminal=0，会切断跨 episode 的 advantage 递推。
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # for distillation
    def generator(self):
        """蒸馏训练的数据生成器。

        每次 yield 一个时间步上的所有环境数据：
            observations[i]:            (E, *obs_shape)
            privileged_observations:    (E, *privileged_obs_shape)
            actions[i]:                 (E, *actions_shape)
            privileged_actions[i]:      (E, *actions_shape)
            dones[i]:                   (E, 1)
        """
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield self.observations[i], privileged_observations, self.actions[i], self.privileged_actions[
                i
            ], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """前馈网络 PPO 的 mini-batch 生成器。

        数据变化可以理解成：

            原始 rollout:
                observations: (T, E, Oa)
                actions:      (T, E, A)
                returns:      (T, E, 1)

            flatten(0, 1) 后:
                observations: (T * E, Oa)
                actions:      (T * E, A)
                returns:      (T * E, 1)

            随机索引 batch_idx 后:
                obs_batch:    (N, Oa)
                actions_batch:(N, A)
                returns_batch:(N, 1)

        这里会重复 num_epochs 轮；每轮用同一组 indices 分成 num_mini_batches 份。
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        # batch_size = T * E，是本轮 rollout 的总样本数。
        batch_size = self.num_envs * self.num_transitions_per_env
        # mini_batch_size = N，假设 batch_size 能整除 num_mini_batches。
        mini_batch_size = batch_size // num_mini_batches
        # indices: (num_mini_batches * N,)，即 (batch_size,) 的随机排列。
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        # flatten(0, 1) 把时间维 T 和环境维 E 合并成样本维 B=T*E。
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        # old_actions_log_prob/advantages/values/returns: (T*E, 1)。
        # old_mu/old_sigma/actions: (T*E, *actions_shape)。
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For RND
        if self.rnd_state_shape is not None:
            # rnd_state: (T*E, *rnd_state_shape)。
            rnd_state = self.rnd_state.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                # batch_idx: (N,)，从展平后的 T*E 样本里随机取一段。
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                # obs_batch: (N, *obs_shape)，privileged_observations_batch: (N, *privileged_obs_shape)。
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                # actions_batch: (N, *actions_shape)。
                actions_batch = actions[batch_idx]

                # -- For PPO
                # target_values_batch/returns_batch/advantages_batch: (N, 1)。
                # old_actions_log_prob_batch: (N, 1)，旧策略联合 log_prob。
                # old_mu_batch/old_sigma_batch: (N, *actions_shape)。
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # -- For RND
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # yield the mini-batch
                yield obs_batch, privileged_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None, rnd_state_batch

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """循环网络 PPO 的 mini-batch 生成器。

        和前馈网络不同，RNN 需要连续轨迹，不能把 (T,E) 直接随机打散。

        流程可视化：
            1. 原始数据按 (T, E, feature) 存放。
            2. 根据 dones 找到 episode 边界，把每个环境的连续片段切成 trajectory。
            3. 不同 trajectory 长度不一样，split_and_pad_trajectories 会 pad 到同一长度。
            4. yield 的 obs_batch 形状类似 (T_pad, num_traj_in_batch, *obs_shape)，
               masks_batch 标出哪些位置是真数据、哪些位置是 padding。

        注意：actions/returns/advantages 这里仍按 (T, env_batch, ...) 给出，
        hidden state 会取每条 trajectory 的起始 hidden。
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        # padded_obs_trajectories: (T_pad, num_trajectories, *obs_shape)。
        # trajectory_masks: (T_pad, num_trajectories)，True/1 表示非 padding 的有效时间步。
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        if self.rnd_state_shape is not None:
            # padded_rnd_state_trajectories: (T_pad, num_trajectories, *rnd_state_shape)。
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        # recurrent 版本按环境维切 batch：每个 mini-batch 包含一段 env id 范围。
        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                # last_was_done[t,e] 表示 (t,e) 是否是一条 trajectory 的起点。
                # t=0 永远是起点；其他时刻如果上一时刻 done，则当前时刻是新 episode 起点。
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                # 当前 env 范围内共有多少条 trajectory。
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # masks_batch: (T_pad, trajectories_batch_size)。
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                # obs_batch: (T_pad, trajectories_batch_size, *obs_shape)。
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                # 这些张量仍保持原始时间维 T 和当前环境子集 env_batch：
                # actions_batch: (T, env_batch, *actions_shape)
                # returns/advantages/values/log_prob: (T, env_batch, 1)
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                # hidden state 只需要每条 trajectory 起点的状态，作为 RNN unroll 的初始状态。
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                # GRU 只有一个 hidden tensor；LSTM 有 (h, c)，所以这里恢复成各自期望的结构。
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, privileged_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch, rnd_state_batch

                first_traj = last_traj
