import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import math
from typing import Optional

from rsl_rl.utils import resolve_nn_activation


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    生成用于 Transformer 的因果掩码。
    True 表示需要遮蔽的未来信息，False 表示可见的过去信息。
    """
    # 【修改】：直接指定 dtype=torch.bool，彻底抛弃后置的 .bool() 转换
    mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1
    )
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 形状变为 (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状应为 (num_envs, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :] # 添加位置编码
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask: Optional[torch.Tensor] = None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)
        self.linear = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, attn_mask: Optional[torch.Tensor] = None): # 增加默认值 None
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2) 
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2) 
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2) 

        context, attn = self.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) 
        output = self.linear(context)
        return self.layer_norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(PoswiseFeedForwardNet, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        # inputs : [batch_size, len_q, d_model]
        residual = inputs 
        output = self.gelu(self.linear1(inputs))
        output = self.linear2(output)
        
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self,d_model, n_heads, dim_feedforward=256):
        super(EncoderLayer, self).__init__()
        d_k = d_v = d_model // n_heads
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=dim_feedforward)

    def forward(self, Q, K, V, attn_mask: Optional[torch.Tensor] = None):
        enc_outputs, attn = self.enc_self_attn(Q, K, V, attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # 包含残差连接和 LayerNorm
        return enc_outputs, attn


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, x):
        # x: [batch_size, len_q, d_model]
        # 在序列维度(len_q)上做max pooling，得到每个d_model维度的最大值
        # 输出: [batch_size, d_model]
        return torch.max(x, dim=1).values

class Transformer(nn.Module):
    def __init__(self, 
                 num_actor_obs, 
                 num_mimic_obs, 
                 num_state_his, 
                 dim_per_mimic, 
                 dim_per_state, 
                 num_actions, 
                 d_model, 
                 nhead, 
                 num_layers, 
                 dim_feedforward):
        super(Transformer, self).__init__()
        # Policy
        self.len_cmd_enc = num_mimic_obs
        self.len_sta_enc = num_state_his
        self.dim_per_mimic = dim_per_mimic
        self.dim_per_state = dim_per_state
        all_cmd_dim = num_mimic_obs * dim_per_mimic
        all_sta_dim = num_state_his * dim_per_state
        assert all_cmd_dim + all_sta_dim == num_actor_obs, "命令和状态的总维度必须等于actor的输入维度"

        self.d_model = d_model
        self.in_features = num_actor_obs

        # command encoder
        self.cmd_input_proj1 = nn.Linear(dim_per_mimic, 256)
        self.cmd_input_proj2 = nn.Linear(256, d_model)
        self.cmd_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cmd_norm1 = nn.LayerNorm(d_model)
        self.cmd_mha_layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])
        self.cmd_output_proj1 = nn.Linear(d_model, 256)
        self.cmd_output_proj2 = nn.Linear(256, d_model)

        # state encoder
        self.sta_input_proj1 = nn.Linear(dim_per_state, 256)
        self.sta_input_proj2 = nn.Linear(256, d_model)
        self.sta_norm1 = nn.LayerNorm(d_model)
        self.sta_self_attn_layer = EncoderLayer(d_model, nhead, dim_feedforward)
        self.sta_cross_mha_layer = EncoderLayer(d_model, nhead, dim_feedforward)
        self.sta_norm2 = nn.LayerNorm(d_model)
        # position encoding
        self.pos_emb = PositionalEncoding(d_model)

        # actor mlp
        self.actor_linear1 = nn.Linear(d_model + dim_per_state, 256)
        self.actor_linear2 = nn.Linear(256, 256)
        self.actor_linear3 = nn.Linear(256, num_actions)

    def forward(self, x):
        # x: [batch_size, len_cmd_enc * dim_per_mimic + len_sta_enc * dim_per_state]
        batch_size = x.size(0)
        cmd_dim = self.len_cmd_enc * self.dim_per_mimic
        sta_dim = self.len_sta_enc * self.dim_per_state

        cmd_flat, state_flat = torch.split(x, [cmd_dim, sta_dim], dim=-1)
        cmd = cmd_flat.view(batch_size, self.len_cmd_enc, self.dim_per_mimic) # cmd.shape: [batch_size, len_cmd_enc, dim_per_mimic]
        state = state_flat.view(batch_size, self.len_sta_enc, self.dim_per_state) # state.shape: [batch_size, len_sta_enc, dim_per_state]
        cur_state = state[:,-1,:]

        # command encoding
        cmd = F.gelu(self.cmd_input_proj1(cmd)) # cmd.shape: [batch_size, len_cmd_enc, 256]
        cmd = self.cmd_input_proj2(cmd) * math.sqrt(self.d_model) # cmd.shape: [batch_size, len_cmd_enc, d_model]
        cmd_cls = self.cmd_cls_token.expand(batch_size, -1, -1)
        cmd = torch.cat([cmd_cls, cmd], dim=1) # cmd.shape: [batch_size, len_cmd_enc+1, d_model]
        cmd = self.pos_emb(cmd) # cmd.shape: [batch_size, len_cmd_enc, d_model]
        cmd = self.cmd_norm1(cmd) # cmd.shape: [batch_size, len_cmd_enc, d_model]
        for mha_layer in self.cmd_mha_layers:
            cmd, _ = mha_layer(cmd, cmd, cmd, attn_mask=None) # cmd.shape: [batch_size, len_cmd_enc, d_model]
        cmd_enc_output = cmd[:, 0, :] # CLS token 聚合结果: [batch_size, d_model]
        cmd_enc_output = F.gelu(self.cmd_output_proj1(cmd_enc_output)) # cmd_enc_output.shape: [batch_size, 256]
        cmd_enc_output = self.cmd_output_proj2(cmd_enc_output) # cmd_enc_output.shape: [batch_size, d_model]
        cmd_query = cmd_enc_output # cmd_query.shape: [batch_size, d_model]

        # state encoding
        state = F.gelu(self.sta_input_proj1(state)) # state.shape: [batch_size, len_sta_enc, 256]
        state = self.sta_input_proj2(state) * math.sqrt(self.d_model) # state.shape: [batch_size, len_sta_enc, d_model]
        state = self.pos_emb(state) # state.shape: [batch_size, len_sta_enc, d_model]
        state = self.sta_norm1(state) # state.shape: [batch_size, len_sta_enc, d_model]
        sta_causal_mask = generate_causal_mask(self.len_sta_enc, state.device)
        state, _ = self.sta_self_attn_layer(state, state, state, attn_mask=sta_causal_mask) # state.shape: [batch_size, len_sta_enc, d_model]
        state, _ = self.sta_cross_mha_layer(cmd_query.unsqueeze(1), state, state, attn_mask=None) # state.shape: [batch_size, 1, d_model]
        state = state.squeeze(1) # state.shape: [batch_size, d_model]
        ut = self.sta_norm2(state)

        # actor mlp
        actor_obs = torch.cat([cur_state, ut], dim=-1) # actor_obs.shape: [batch_size, dim_per_state+d_model]
        h = F.elu(self.actor_linear1(actor_obs))  # h.haspe:[batch_size, 256]
        h = F.elu(self.actor_linear2(h))  # h.haspe:[batch_size, 256]
        actions = self.actor_linear3(h)  # h.haspe:[batch_size, num_actions]

        return actions

    def __getitem__(self, idx):
        """
        专属 Hack：欺骗 rsl_rl 的 ONNX 导出器。
        当导出器尝试访问 self.actor[0].in_features 时，
        让 self.actor[0] 直接返回 Transformer 实例本身，
        随后它就能顺利读取到上面定义的 self.in_features。
        """
        if idx == 0:
            return self
        raise IndexError("Transformer dummy indexing out of bounds")


class TransformerActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_mimic_obs,  # mimic帧数的数量
        num_state_his,  # 历史状态的数量
        dim_per_mimic,  # 每个模仿对象的观测维度
        dim_per_state,  # 每个状态的观测维度
        critic_hidden_dims=[512, 256, 128],
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.0,
        activation="relu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "TransformerActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.d_model = d_model
        mlp_input_dim_c = num_critic_obs

        self.actor = Transformer(num_actor_obs,
                                 num_mimic_obs, 
                                 num_state_his, 
                                 dim_per_mimic, 
                                 dim_per_state, 
                                 num_actions, 
                                 d_model, 
                                 nhead, 
                                 num_layers, 
                                 dim_feedforward)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(
                        critic_hidden_dims[layer_index],
                        critic_hidden_dims[layer_index + 1],
                    )
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )
        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
