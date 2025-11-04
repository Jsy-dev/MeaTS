import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from utils.model_utils import LearnablePositionalEncoding, Proj, AdaLayerNorm, Transpose, GELU2, series_decomp, Embedding, FreqProj, Proj, FreqEmbedding
from utils.RevIN import RevIN


class FourierLayer(nn.Module):
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


class FullAttention(nn.Module):
       #The code wil be avilable when this paper accepted by TII

        return y, att, q, k


class FeatureFusion(nn.Module):
    def __init__(
            self,
            n_embd,
            n_head,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            n_layer=1,
            type='random',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[FeatureFusionBlock(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        ) for _ in range(n_layer)])

        self.ConvFusion = nn.Linear(n_embd, n_embd)

    def forward(self, x1, x2, padding_masks=None, label_emb=None):
        memory_out = (x1+x2)/2
        for block_idx in range(len(self.blocks)):
            x1, x2, memory_out, att1, att2 = self.blocks[block_idx](x1, x2, memory=memory_out)
        x = x1+x2
        x = self.ConvFusion(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 redius=True,
                 ):
        super().__init__()

        self.ln2 = nn.LayerNorm(n_embd)
        self.redius = redius
        self.attn = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        if self.redius:
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )

    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att, k, q = self.attn(x, mask=mask)
        x = x + a

        if self.redius:
            x = x + self.mlp(self.ln2(x))
        return x, att


class Encoder(nn.Module):
    def __init__(
            self,
            n_layer=14,
            n_embd=1024,
            n_head=16,
            attn_pdrop=0.,
            resid_pdrop=0.,
            mlp_hidden_times=4,
            block_activate='GELU',
            max_len=1024,
            position_n=64,
            position_invers=0,
            redius=False
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
            redius=redius,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, att = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)

        return x


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super(Flatten_Head, self).__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            self.norm = nn.LayerNorm(target_window)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class Mformer(nn.Module):
    def __init__(
            self,
            n_feat,
            n_channel,
            n_layer_enc=5,
            n_layer_dec=14,
            n_embd=1024,
            n_heads=16,
            n_decom=1,
            n_proj=2,
            pred_len=96,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            mlp_hidden_times=4,
            block_activate='GELU',
            head_dropout=0,
            max_len=2048,
            conv_params=None,
            patch_size=8,
            patch_stride=4,
            noise_type='random',
            type_embedding=0,
            **kwargs
    ):
        super().__init__()
        if patch_size != patch_stride:
            self.output_len = [n_feat + (patch_size - patch_stride)]
        else:
            self.output_len = [n_feat]

        self.output_len.append(
            math.floor((self.output_len[-1] + 2 * 0 - 1 * (patch_size - 1) - 1) / patch_stride + 1))
        self.n_embd = n_embd * self.output_len[-1]
        print('latent_num', self.n_embd)

        if patch_size != patch_stride:
            self.output_len = [n_channel + (patch_size - patch_stride)]
        else:
            self.output_len = [n_channel]

        self.output_len.append(
            math.floor((self.output_len[-1] + 2 * 0 - 1 * (patch_size - 1) - 1) / patch_stride + 1))
        if type_embedding == 0:
            self.n_embd_pre = n_embd * self.output_len[-1]
        elif type_embedding == 1:
            if patch_stride != patch_size:
                patch_num = (n_channel // patch_stride)-1
            else:
                patch_num = (n_channel // patch_stride)
            self.n_embd_pre = n_embd * patch_num
        print('latent_num_pre', self.n_embd_pre)

        self.embedding = Embedding(out_dim=n_embd, patch_size=patch_size, patch_stride=patch_stride, num=n_proj, resid_pdrop=resid_pdrop, type=type_embedding)

        self.n_layer_dec = n_layer_dec

        print(n_layer_enc, self.n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times)

        print(n_channel, n_feat, self.n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times)

        self.revin_layer = RevIN(n_feat, affine=0, subtract_last=0)
        self.proj_flatten = Flatten_Head(individual=False, n_vars=n_feat, nf=self.n_embd_pre, target_window=pred_len, head_dropout=head_dropout)
        self.encoder_pre = Encoder(n_layer_enc, self.n_embd_pre, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times,
                                    block_activate, max_len=n_feat, position_n=self.n_embd_pre, position_invers=0, redius=True)
        self.ELV = FreqProj(n_feat*n_channel, n_feat*n_channel)
        self.fusion = Proj(n_channel*2, n_channel, num=1)

    def forecasting(self, emb, t=None, padding_masks=None):
        B1, C1, N1, D1 = emb.shape
        emb = emb.reshape(B1, C1, N1 * D1)
        if t is None:
            t = torch.zeros(emb.shape[0], device=emb.device)
        enc_cond_intra = self.encoder_pre(emb, t, padding_masks=padding_masks)

        FF = enc_cond_intra.reshape(B1, C1, N1, D1)
        out = self.proj_flatten(FF)

        return out

    def forward(self, input, t=None, padding_masks=None, return_res=False, ox=None):
        B, N, C = input.shape
        if padding_masks is not None:
            means = torch.sum(input, dim=1) / torch.sum(padding_masks == 1, dim=1)
            means = means.unsqueeze(1).detach()
            input = input - means
            input = input.masked_fill(padding_masks == 0, 0)
            stdev = torch.sqrt(torch.sum(input * input, dim=1) /
                               torch.sum(padding_masks == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            input /= stdev
        else:
            input = self.revin_layer(input, 'norm')
        im_x = self.ELV(input.reshape(B, N*C)).reshape(B, N, C)
        fusion_x = self.fusion(torch.cat((input, im_x), dim=1).permute(0, 2, 1)).reshape(B, N, C)
        conmbi = input+fusion_x

        emb = self.embedding(conmbi)

        out = self.forecasting(emb, t, padding_masks=padding_masks)
        out = out.permute(0, 2, 1)
        if padding_masks is not None:
            out = out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(1, N, 1))
            out = out + \
                      (means[:, 0, :].unsqueeze(1).repeat(1, N, 1))
        else:
            out = self.revin_layer(out, 'denorm')

        return out, self.revin_layer(conmbi, 'denorm')


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.n_feat = configs.n_feat
        self.seq_len = configs.seq_len
        self.n_embd = configs.n_embd
        self.n_layer_enc = configs.n_layer_enc
        self.n_layer_dec = configs.n_layer_dec
        self.n_heads = configs.n_heads
        self.n_proj = configs.n_proj
        self.attn_pdrop = configs.attn_pdrop
        self.resid_pdrop = configs.resid_pdrop
        self.pred_len = configs.pred_len
        self.mlp_hidden_times = configs.mlp_hidden_times
        self.block_activate = configs.block_activate
        self.max_len = configs.max_len
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.head_dropout = configs.head_dropout
        self.model = Mformer(n_feat=self.n_feat, n_channel=self.seq_len, n_layer_dec=self.n_layer_dec,
                             n_layer_enc=self.n_layer_enc, n_heads=self.n_heads, n_proj=self.n_proj, n_embd=self.n_embd, attn_pdrop=self.attn_pdrop,
                             patch_stride=self.patch_stride, patch_size=self.patch_size, resid_pdrop=self.resid_pdrop, pred_len=self.pred_len, head_dropout=self.head_dropout)

    def forward(self, x, te=None):
        x, im_x = self.model(x)

        return x, im_x


if __name__ == '__main__':

    pass
