import math
import scipy
import torch
import torch.nn.functional as F

from torch import nn, einsum
from functools import partial
from einops import rearrange, reduce
from scipy.fftpack import next_fast_len
import torch.fft as fft


def exists(x):
    """
    Check if the input is not None.

    Args:
        x: The input to check.

    Returns:
        bool: True if the input is not None, False otherwise.
    """
    return x is not None

def default(val, d):
    """
    Return the value if it exists, otherwise return the default value.

    Args:
        val: The value to check.
        d: The default value or a callable that returns the default value.

    Returns:
        The value if it exists, otherwise the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    """
    Return the input tensor unchanged.

    Args:
        t: The input tensor.
        *args: Additional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        The input tensor unchanged.
    """
    return t

def extract(a, t, x_shape):
    """
    Extracts values from tensor `a` at indices specified by tensor `t` and reshapes the result.
    Args:
        a (torch.Tensor): The input tensor from which values are extracted.
        t (torch.Tensor): The tensor containing indices to extract from `a`.
        x_shape (tuple): The shape of the tensor `x` which determines the final shape of the output.
    Returns:
        torch.Tensor: A tensor containing the extracted values, reshaped to match the shape of `x` except for the first dimension.
    """

    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cond_fn(x, t, classifier=None, y=None, classifier_scale=1.):
    """
    Compute the gradient of the classifier's log probabilities with respect to the input.

    Args:
        classifier (nn.Module): The classifier model used to compute logits.
        x (torch.Tensor): The input tensor for which gradients are computed.
        t (torch.Tensor): The time step tensor.
        y (torch.Tensor, optional): The target labels tensor. Must not be None.
        classifier_scale (float, optional): Scaling factor for the gradients. Default is 1.

    Returns:
        torch.Tensor: The gradient of the selected log probabilities with respect to the input tensor, scaled by classifier_scale.
    """
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

# normalization functions

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding module.

    This module generates sinusoidal positional embeddings for input tensors.
    The embeddings are computed using sine and cosine functions with different frequencies.

    Attributes:
        dim (int): The dimension of the positional embeddings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 1
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos(), emb.cos()), dim=-1)[:,:self.dim]
        return emb


# learnable positional embeds

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding module.

    This module generates learnable positional embeddings for input tensors.
    The embeddings are learned during training and can adapt to the specific task.

    Attributes:
        d_model (int): The dimension of the positional embeddings.
        dropout (float): The dropout rate applied to the embeddings.
        max_len (int): The maximum length of the input sequences.
    """
    def __init__(self, d_model, dropout=0.1, max_len=1024, invers=0):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        if invers==1:
            self.pe = nn.Parameter(torch.empty(1, d_model, max_len))  # requires_grad automatically set to True
        else:
            self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # print(x.shape)
        x = x + self.pe
        if torch.isnan(self.pe).any().item():
            print('self.pe')
        return self.dropout(x)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class FreqEmbedding(nn.Module):
    def __init__(self, out_dim=128, final_dim=6, patch_size=1, patch_stride=1, num=2, resid_pdrop=0., type=0,alig_dim=None):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.type = type
        self.out_dim = out_dim  # 固定为128
        self.final_dim = final_dim  # 固定为6

        # 核心：频域处理后通过卷积将维度压缩到final_dim
        self.freq_compress = nn.Conv1d(out_dim, final_dim, kernel_size=1, stride=1)

        if type == 0:
            self.sequential = nn.Sequential(
                # 卷积层输出通道设为out_dim=128
                nn.Conv1d(in_channels=1, out_channels=out_dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(p=resid_pdrop),
            )
            for _ in range(num):
                self.sequential.append(nn.Conv1d(out_dim, out_dim, 1, stride=1))
                self.sequential.append(nn.GELU())
                self.sequential.append(nn.BatchNorm1d(out_dim))
                self.sequential.append(nn.Dropout(p=resid_pdrop))
        else:
            # 若使用type=1，需对应调整线性层维度
            self.sequential = nn.Sequential(
                nn.Linear(patch_size, out_dim),
                nn.ReLU(),
            )
            for _ in range(num):
                self.sequential.append(nn.Linear(out_dim, out_dim))
                self.sequential.append(nn.ReLU())
            self.final_linear = nn.Linear(out_dim, final_dim)

    def forward(self, x):
        # 原始输入：(200, 12, 170) → (B, D, N)
        B, D, N = x.shape  # B=200, D=12, N=170（目标需保留N=170）

        if self.type == 0:
            # 1. 保留原始序列长度N=170（不做padding，避免破坏N）
            x_reshape = x  # 形状：(200, 12, 170)

            # 2. 时域→频域转换（对N维度做FFT）
            x_freq = fft.rfft(x_reshape, dim=-1)  # 频域形状：(200, 12, fft_dim)
            # fft_dim = (170//2)+1 = 86（因170是偶数：170//2=85 → 85+1=86）
            x_mag = torch.abs(x_freq)  # 幅度谱：(200, 12, 86)

            # 3. 调整形状适配卷积（保留N相关维度）
            # 增加通道维度 → (200, 1, 12, 86)
            x_mag = x_mag.unsqueeze(1)
            # 重塑为：(200*12, 1, 86) → 确保后续处理后可恢复N
            batch_size = B * D
            fft_dim = x_mag.shape[-1]  # 86
            x_mag = x_mag.reshape(batch_size, 1, fft_dim)  # (2400, 1, 86)

            # 4. 频域卷积处理（输出out_dim=128）
            x_processed = self.sequential(x_mag)  # 形状：(2400, 128, 86)

            # 5. 压缩到final_dim=6（通过1x1卷积）
            x_compress = self.freq_compress(x_processed)  # 形状：(2400, 6, 86)

            # 6. 恢复维度并调整顺序至目标形状
            # 重塑为：(200, 12, 6, 86) → 对D维度聚合（平均）
            x_restore = x_compress.reshape(B, D, self.final_dim, -1)  # (200, 12, 6, 86)
            x_agg = x_restore.mean(dim=1)  # 对D=12平均 → (200, 6, 86)

            # 7. 扩展维度至目标out_dim=128（通过重复或投影）
            # 方法1：简单重复（若允许）→ (200, 128, 6, 86)
            x_expand = x_agg.unsqueeze(1).repeat(1, self.out_dim, 1, 1)  # (200, 128, 6, 86)
            # 方法2：通过线性层投影（更合理）→ 此处用方法1快速匹配维度

            # 8. 调整维度顺序并截断/对齐至N=170
            # 目标顺序：(B, N, out_dim, final_dim) → (200, 170, 128, 6)
            # 因频域维度86 < 170，通过重复扩展至170（或根据业务用插值）
            x_out = x_expand.permute(0, 3, 1, 2)  # (200, 86, 128, 6)
            x_out = x_out.repeat(1, 2, 1, 1)[:, :170, :, :]  # 重复后截断至170

            # 最终形状：(200, 170, 128, 6)

        elif self.type == 1:
            # 线性层版本（确保维度匹配）
            x_unfold = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)  # 保留N相关
            x_processed = self.sequential(x_unfold)  # 中间形状
            x_out = self.final_linear(x_processed)  # 压缩至final_dim=6
            # 调整维度顺序至(200, 170, 128, 6)（需根据中间形状微调）

        return x_out

class Embedding(nn.Module):
    def __init__(self, out_dim, patch_size, patch_stride, num=2, resid_pdrop=0., type=0, alig_dim=None):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.type = type
        self.alig_dim =alig_dim
        if alig_dim is not None:
            self.proj = nn.Linear(alig_dim[0], alig_dim[1])
        if type==0:
            self.sequential = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=out_dim, kernel_size=patch_size, stride=patch_stride),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(p=resid_pdrop),
            )
            for _ in range(num):
                self.sequential.append(nn.Conv1d(out_dim, out_dim, 1, stride=1))
                self.sequential.append(nn.GELU())
                self.sequential.append(nn.BatchNorm1d(out_dim))
                self.sequential.append(nn.Dropout(p=resid_pdrop))
        else:
            self.sequential = nn.Sequential(
                nn.Linear(patch_size, out_dim),
                # nn.ReLU(),
            )
            for _ in range(num):
                self.sequential.append(nn.Linear(out_dim, out_dim))
                # self.sequential.append(nn.Dropout(0.1))
                # self.sequential.append(nn.ReLU())

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(-2)
        B, M, D, N = x.shape
        if self.type == 0:
            x = x.reshape(B * M, D, N)
            if self.patch_size != self.patch_stride:
                pad_len = self.patch_size - self.patch_stride
                pad = x[:, :, -1:].repeat(1, 1, pad_len)
                x = torch.cat([x, pad], dim=-1)
            x = self.sequential(x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            if self.alig_dim is not None:
                x = self.proj(x)

        elif self.type == 1:
            x = x.reshape(B, M, 1, -1).squeeze(-2)
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
            x = self.sequential(x)
        return x


# class Embedding(nn.Module):
#     def __init__(self, out_dim, feature_dim, patch_size, patch_stride, patch_num, num=2, resid_pdrop=0., type=0,  alig_dim=None):
#         super().__init__()
#         self.patch_size = patch_size
#         self.patch_stride = patch_stride
#         self.type = type
#         self.alig_dim =alig_dim
#         self.patch_num = patch_num
#
#         self.in_feature_dim = feature_dim
#         self.in_sequence_dim = patch_num*out_dim
#
#
#         if alig_dim is not None:
#             self.proj = nn.Linear(alig_dim[0], alig_dim[1])
#         if type==0:
#             self.sequential = nn.Sequential(
#                 nn.Conv1d(in_channels=1, out_channels=out_dim, kernel_size=patch_size, stride=patch_stride),
#                 nn.BatchNorm1d(out_dim),
#                 nn.GELU(),
#                 nn.Dropout(p=resid_pdrop),
#             )
#             for _ in range(num):
#                 self.sequential.append(nn.Conv1d(out_dim, out_dim, 1, stride=1))
#                 self.sequential.append(nn.GELU())
#                 self.sequential.append(nn.BatchNorm1d(out_dim))
#                 self.sequential.append(nn.Dropout(p=resid_pdrop))
#         else:
#             self.sequential = nn.Sequential(
#                 nn.Linear(patch_size, out_dim),
#                 nn.ReLU(),
#             )
#             for _ in range(num):
#                 self.sequential.append(nn.Linear(out_dim, out_dim))
#                 # self.sequential.append(nn.Dropout(0.1))
#                 self.sequential.append(nn.ReLU())
#
#         # self.Cv_feature_dim = nn.Conv1d(in_channels=self.in_feature_dim, out_channels=self.in_feature_dim , kernel_size=1, stride=1)
#         # self.Cv_sequence_dim = nn.Conv1d(in_channels=self.in_sequence_dim, out_channels=self.in_sequence_dim , kernel_size=1, stride=1)
#         # self.act_feature_dim = nn.GELU()
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = x.unsqueeze(-2)
#         B, M, D, N = x.shape
#         if self.type == 0:
#             x = x.reshape(B * M, D, N)
#             if self.patch_size != self.patch_stride:
#                 pad_len = self.patch_size - self.patch_stride
#                 pad = x[:, :, -1:].repeat(1, 1, pad_len)
#                 x = torch.cat([x, pad], dim=-1)
#             x = self.sequential(x)
#             _, D_, N_ = x.shape
#             # x = x.reshape(B, M, D_*N_)
#             # x = self.Cv_feature_dim(x)
#             # x = self.act_feature_dim(x)
#             x = x.reshape(B, M, D_, N_)
#
#             if self.alig_dim is not None:
#                 x = self.proj(x)
#
#
#         elif self.type == 1:
#             x = x.reshape(B, M, 1, -1).squeeze(-2)
#             x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
#             x = self.sequential(x)
#         return x

# class Proj(nn.Module):
#     def __init__(self, in_dim, out_dim, num=1, resid_pdrop=0.):
#         super().__init__()
#         self.sequential = nn.Sequential(
#             Transpose(shape=(1, 2)),
#             nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
#             nn.Dropout(p=resid_pdrop),
#             nn.GELU(),
#         )
#         for _ in range(num):
#             self.sequential.append(nn.Conv1d(out_dim, out_dim, 1, stride=1, padding=0))
#             self.sequential.append(nn.Dropout(p=resid_pdrop))
#             self.sequential.append(nn.GELU())
#
#     def forward(self, x):
#         return self.sequential(x).transpose(1, 2)

class Proj(nn.Module):
    def __init__(self, in_dim, out_dim, num=1, resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(p=resid_pdrop),
            nn.GELU(),
        )
        for _ in range(num):
            self.sequential.append(nn.Linear(out_dim, out_dim))
            self.sequential.append(nn.Dropout(p=resid_pdrop))
            self.sequential.append(nn.GELU())

    def forward(self, x):
        return self.sequential(x)


class FreqProj(nn.Module):
    def __init__(self, in_dim, out_dim, num=1, resid_pdrop=0.):
        super().__init__()
        # 计算FFT后的维度
        self.fft_dim = (in_dim // 2) + 1  # rfft后的典型维度计算公式

        # 频域线性变换层，输入维度调整为fft_dim
        self.freq_sequential = nn.Sequential(
            nn.Linear(self.fft_dim, out_dim),
            nn.Dropout(p=resid_pdrop),
            nn.GELU(),
        )

        for _ in range(num):
            self.freq_sequential.append(nn.Linear(out_dim, out_dim))
            self.freq_sequential.append(nn.Dropout(p=resid_pdrop))
            self.freq_sequential.append(nn.GELU())

    def forward(self, x):
        # 1. 时域信号转换到频域
        x_freq = torch.fft.rfft(x, dim=-1)

        # 2. 分离实部和虚部
        real_part = x_freq.real
        imag_part = x_freq.imag

        # 3. 分别对实部和虚部应用频域变换
        real_transformed = self.freq_sequential(real_part)
        imag_transformed = self.freq_sequential(imag_part)

        # 4. 重新组合复数
        x_freq_transformed = torch.complex(real_transformed, imag_transformed)

        # 5. 逆变换回时域
        x_time = torch.fft.irfft(x_freq_transformed, n=x.size(-1), dim=-1)

        # 6. 残差连接
        return x_time

class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Dropout(p=resid_pdrop),
            nn.Conv1d(out_dim, out_dim, 1, stride=1, padding=0),
            nn.Dropout(p=resid_pdrop),
            nn.Conv1d(out_dim, out_dim, 1, stride=1, padding=0),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)


class Transformer_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=1, padding=0),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd), out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, padding=1),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd,  kernel_size=3, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x)
    

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x
    

class AdaInsNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1,-2) * (1 + scale) + shift
        return x