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

class EMAVectorQuantizer(nn.Module):
    """单层基于 EMA (指数移动平均) 的量化器"""
    def __init__(self, num_codes, d_model, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.d_model = d_model
        self.decay = decay
        self.eps = eps

        # 密码本：不参与梯度反向传播！
        embed = torch.randn(num_codes, d_model)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(num_codes))
        self.register_buffer('embed_sum', embed.clone())

    def forward(self, x):
        # 计算欧氏距离: (x-y)^2 = x^2 - 2xy + y^2
        # x shape: [batch_size, d_model]
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.embed.t()
            + self.embed.pow(2).sum(1, keepdim=True).t()
        )
        
        # 寻找最近的 Code 索引
        _, embed_ind = dist.min(1) # embed_ind shape: [batch_size]
        quantized = F.embedding(embed_ind, self.embed) # quantized shape: [batch_size, d_model]
        
        # EMA 在线更新逻辑 (仅在训练模式下执行)
        if self.training:
            embed_onehot = F.one_hot(embed_ind, self.num_codes).type(x.dtype) # shape: [batch_size, num_codes]
            
            # 统计频率 (N_i) shape: [num_codes]
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot.sum(0), alpha=1 - self.decay
            ) 
            # 统计向量和 (m_i) shape: [num_codes, d_model]
            embed_sum_ze = embed_onehot.t() @ x # shape: [num_codes, d_model]
            self.embed_sum.data.mul_(self.decay).add_(embed_sum_ze, alpha=1 - self.decay)
            
            # 计算新的均值并覆盖 Codebook
            n = self.cluster_size.sum() # 这个 n 是一个标量，表示所有样本的总数（加上 eps 平滑项）
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
            )
            embed_normalized = self.embed_sum / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            
        return quantized, embed_ind

class ResidualVectorQuantizer(nn.Module):
    """残差量化器 (RVQ)"""
    def __init__(self, num_layers=4, num_codes=512, d_model=128, commitment_weight=0.25):
        super().__init__()
        self.num_layers = num_layers
        self.commitment_weight = commitment_weight
        
        # 实例化 4 层独立的 EMA 字典
        self.quantizers = nn.ModuleList([
            EMAVectorQuantizer(num_codes, d_model) for _ in range(num_layers)
        ])

    def forward(self, z_e):
        # z_e shape: [batch_size, d_model]
        z_q = torch.zeros_like(z_e)
        residual = z_e
        
        all_indices = []
        vq_loss = torch.zeros((), device=z_e.device, dtype=z_e.dtype)
        
        # Quantizer Dropout 逻辑：训练时随机截断，推理时全开
        if self.training:
            num_active_layers = torch.randint(1, self.num_layers + 1, (1,)).item()
        else:
            num_active_layers = self.num_layers

        for i, quantizer in enumerate(self.quantizers):
            # 去字典里找残差
            quantized, indices = quantizer(residual)
            
            # 只有在激活层内的特征才会被累加
            if i < num_active_layers:
                z_q = z_q + quantized
                # 计算 Commitment Loss (让 Encoder 吐出的 z_e 靠近选出的 z_q)
                # 注意：因为 Codebook 是 EMA 更新的，所以这里只需要单向 Loss
                vq_loss += F.mse_loss(quantized.detach(), residual) * self.commitment_weight
            
            # 无论是否累加到输出，下一层的目标永远是拟合上一层的残差
            residual = residual - quantized.detach()
            all_indices.append(indices)

        # 致命保留：STE (直通估计器) 魔法！
        # 让 z_q 带着 z_e 的梯度流回去，否则 Encoder 无法训练
        z_q = z_e + (z_q - z_e).detach()
        
        # indices_tensor shape: [batch_size, num_layers]
        indices_tensor = torch.stack(all_indices, dim=-1)
        
        return z_q, vq_loss, indices_tensor


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

        # ===== 【新增代码】：植入 RVQ 瓶颈 =====
        self.rvq = ResidualVectorQuantizer(
            num_layers=4, 
            num_codes=512, 
            d_model=d_model, 
            commitment_weight=0.25
        )
        # 记录当前的 vq_loss，供外部的 compute_loss 调用
        self.vq_loss = torch.tensor(0.0)
        self.recon_loss = torch.tensor(0.0)

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

        # =========================================================
        # ===== 【新增】：Stage 1 专属 Cmd Decoder (序列重构器) =====
        # =========================================================
        # 1. 可学习的 Query Tokens：长度与原始的 cmd 序列长度完全一致
        self.cmd_dec_queries = nn.Parameter(torch.zeros(1, self.len_cmd_enc, d_model))
        nn.init.normal_(self.cmd_dec_queries, std=0.02) # 标准的 Transformer 初始化
        
        # 2. Decoder 的注意力层：
        #    - Self-Attention: 让生成的序列内部平滑连续
        #    - Cross-Attention: 负责去 $z_q$ 里面“提取/解压缩”宏观动作信息
        self.cmd_dec_self_attn = EncoderLayer(d_model, nhead, dim_feedforward)
        self.cmd_dec_cross_attn = EncoderLayer(d_model, nhead, dim_feedforward)
        
        # 3. 输出映射层：把 d_model 维度的特征还原成具体的物理量（如关节位置/速度）
        self.cmd_dec_output_proj = nn.Linear(d_model, dim_per_mimic)
        # =========================================================

    def forward(self, x):
        # x: [batch_size, len_cmd_enc * dim_per_mimic + len_sta_enc * dim_per_state]
        batch_size = x.size(0)
        cmd_dim = self.len_cmd_enc * self.dim_per_mimic
        sta_dim = self.len_sta_enc * self.dim_per_state

        cmd_flat, state_flat = torch.split(x, [cmd_dim, sta_dim], dim=-1)
        cmd = cmd_flat.view(batch_size, self.len_cmd_enc, self.dim_per_mimic) # cmd.shape: [batch_size, len_cmd_enc, dim_per_mimic]
        original_cmd = cmd
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
        # cmd_query = cmd_enc_output # cmd_query.shape: [batch_size, d_model]

        # ===== 【修改代码】：量化查表与特征替换 =====
        z_e = cmd_enc_output
        # 送入 RVQ 瓶颈，得到离散拼合的 z_q
        z_q, vq_loss, indices = self.rvq(z_e)
        
        # 将 vq_loss 挂载到实例上，在 rsl_rl 的 PPO loss 计算中把它加进去！
        self.vq_loss = vq_loss 
        
        # 偷天换日：小脑后续接收到的指令不再是丝滑的 z_e，而是阶梯状的 z_q
        cmd_query = z_q 
        # ============================================

        # =========================================================
        # ================== 【核心：Cmd 解码重构】 ==================
        # =========================================================
        
        # A. 铺开 Queries 并加上位置编码
        dec_q = self.cmd_dec_queries.expand(batch_size, -1, -1) # [batch, len_cmd, d_model]
        dec_q = self.pos_emb(dec_q)
        
        # B. Queries 进行 Self-Attention (建立序列内部的时间关系)
        dec_q, _ = self.cmd_dec_self_attn(dec_q, dec_q, dec_q, attn_mask=None)
        
        # C. Queries 去 Cross-Attention 中“读取” z_q
        # 把 z_q 扩展一个维度作为 Memory: [batch_size, 1, d_model]
        memory = z_q.unsqueeze(1) 
        dec_out, _ = self.cmd_dec_cross_attn(dec_q, memory, memory, attn_mask=None)
        
        # D. 映射回原始的物理维度
        cmd_recon = self.cmd_dec_output_proj(dec_out) # [batch_size, len_cmd, dim_per_mimic]

        # =========================================================
        # ================== 【计算 Stage 1 总 Loss】 =================
        # =========================================================
        
        # 计算序列重构误差 (MSE Loss)
        # 注意：这里我们让重构出的序列，去逼近最开始输入的 original_cmd
        self.recon_loss = F.mse_loss(cmd_recon, original_cmd.detach())

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
