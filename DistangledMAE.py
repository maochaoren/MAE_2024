import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from einops import reduce, rearrange, repeat
from RetroMAElike_model import RetroMaskedAutoEncoder

import numpy as np
class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)

class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs


        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)#线性映射
        #print(self.num_freqs)
        #print(output_fft.shape)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)#  out=W*x+B
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)#calculate fan_in_and_fan_out:输出每层各个节点的输入、输出数据个数。
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


class Distangle(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial'):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        self.tfd = nn.ModuleList( #trend feature disentangler,TFD
            [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
        )

        self.sfd = nn.ModuleList(#season feature disentangler,SFD
            [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
        )

    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims

        # conv encoder
        x=self.input_fc(x)
        x = x.transpose(1, 2)  # B x Ch x T
        #print(x.shape)
        x = self.feature_extractor(x)  # B x Co x T

        if tcn_output:
            return x.transpose(1, 2)

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # b t d
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'#在List这一维度求平均，即“平均池化”操作。
        )

        x = x.transpose(1, 2)  # B x T x Co

        season = []
        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]

        return trend, self.repr_dropout(season)#趋势、季节性特征各自的维度为out_dim/2

class DistangleMAE(nn.Module):
    def __init__(self,c_in,d_model,input_len,mask_rate_enc,mask_rate_dec,encoder_depth,enhance_decoding
                  ,alpha=1.0,train_pe=True):        
        super(DistangleMAE,self).__init__()
        self.alpha=alpha
        self.d_model=d_model
        self.input_len=input_len
        self.mask_rate_enc=mask_rate_enc
        self.mask_rate_dec=mask_rate_dec
        self.enhance_decoding=enhance_decoding
        self.c_in=c_in
        self.mask_token=nn.Parameter(torch.zeros(1,1,d_model))

        self.conv_embed=nn.Conv1d(c_in,d_model,kernel_size=3,padding=1,padding_mode='circular')
        #distangle
        self.distangle=Distangle(input_dims=d_model,output_dims=d_model,kernels=[2**i for i in range(0,int(math.log2(input_len))+1)]
                                 ,length=input_len)
        #mae 季节趋势分解，两路分别用mae获得repre,将两路的repre拼接获得时间序列的repre
        self.trend_mae=RetroMaskedAutoEncoder(encoder_depth=encoder_depth,c_in=d_model//2,d_model=d_model//2,input_len=input_len,series_embed_len=1,mask_rate_enc=mask_rate_enc,
            mask_rate_dec=mask_rate_dec,mask_size=1,enhance_decoding=enhance_decoding,alpha=1.0,train_pe=train_pe)
        
        self.season_mae=RetroMaskedAutoEncoder(encoder_depth=encoder_depth,c_in=d_model//2,d_model=d_model//2,input_len=input_len,series_embed_len=1,mask_rate_enc=mask_rate_enc,
            mask_rate_dec=mask_rate_dec,mask_size=1,enhance_decoding=enhance_decoding,alpha=1.0,train_pe=train_pe)

        #预测
        self.pred=nn.Linear(d_model,c_in)
        self.criterion=nn.MSELoss()

    def forward(self,x):
        x_pred=x
        x=self.conv_embed(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
        x_trend,x_season=self.distangle(x)#embedding,distangle
        x_season,loss_s=self.season_mae(x_season)
        x_trend,loss_t=self.trend_mae(x_trend)
        repr=torch.concat([x_trend,x_season],dim=2)

        pred=self.pred(repr)
        loss_total=self.criterion(pred,x_pred)

        print('loss_t:{}'.format(loss_t))
        print('loss_s:{}'.format(loss_s))
        print('loss:{}'.format(loss_total))

        loss=(loss_s+loss_t*self.alpha)/(1.0+self.alpha)+loss_total

        return x_trend,loss

