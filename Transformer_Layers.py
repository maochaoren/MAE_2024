
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Mutihead_Attention,Feed_Forward,FullAttention,AttentionLayer
from RetroMAElike_model import RetroMAE_Encoder,instance_denorm,instance_norm

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

class MultiDecompTransformerEncoderLayer(nn.Module):#autoformer encoder-like
    def __init__(self,c_in,d_model,input_len,window_size,dropout=0.1):
        super(MultiDecompTransformerEncoderLayer, self).__init__()
        self.d_model=d_model
        self.input_len=input_len
        #self.embed_dim=embed_dim
        self.c_in=c_in
        self.mask_token=nn.Parameter(torch.zeros(1,1,d_model))
        
        self.layernorm=nn.LayerNorm(d_model)
        self.attention=Mutihead_Attention(d_model=d_model,dim_k=d_model,dim_v=d_model,requires_mask=False,n_heads=4)
        self.feed_forward=Feed_Forward(input_dim=d_model)
        self.series_decomp1=series_decomp(kernel_size=window_size)
        self.series_decomp2=series_decomp(kernel_size=window_size)
        self.series_decomp3=series_decomp(kernel_size=window_size)

        self.t1=nn.Linear(d_model,d_model,bias=True)
        self.t2=nn.Linear(d_model,d_model,bias=True)

        self.dropout=nn.Dropout(dropout)
        self.activation=F.relu

        self.conv1=nn.Conv1d(in_channels=d_model,out_channels=d_model*4,kernel_size=1,bias=False)
        self.conv2=nn.Conv1d(in_channels=d_model*4,out_channels=d_model,kernel_size=1,bias=False)

        self.projection=nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=3,stride=1,padding=1,padding_mode='circular',bias=False)


    def forward(self,x_s,x_t):
        #s
        #x_s,x_t=self.series_decomp1(input)
        x_s=self.layernorm(x_s)
        x_s_=self.attention(x_s,x_s,x_s,x_s,x_s)
        x_s=x_s+self.dropout(x_s_)
        x_s,trend1=self.series_decomp1(x_s)
        x_s_=x_s

        x_s_=self.dropout(self.activation(self.conv1(x_s_.transpose(-1,1))))
        x_s_=self.dropout(self.conv2(x_s_).transpose(-1,1))
        #x_s_=x_s
        #x_s=self.feed_forward(x_s)+x_s
        x_s,trend2=self.series_decomp2(x_s+x_s_)

        #t
        #x_t=x_t+self.t1(trend1)+self.t2(trend2)
        #x_t=x_t+trend1
        x_t=self.projection(x_t.permute(0,2,1)).transpose(1,2)
        return x_s,x_t
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series (same pad)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
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
        return res, moving_mean #s t

class ResidualDecompEncoderLayer(nn.Module):
    def __init__(self,d_model,n_head,window_size=2+1):
        super(ResidualDecompEncoderLayer,self).__init__()
        self.series_decomp=series_decomp(window_size)
        self.EncoderLayer=EncoderLayer(attention=AttentionLayer(FullAttention(mask_flag=False,output_attention=False),
                                                                d_model=d_model,n_heads=n_head),d_model=d_model)
        self.proj=nn.Linear(d_model,d_model)

    def forward(self,x):
        s,t=self.series_decomp(x)
        t,_=self.EncoderLayer(t)
        #print(t.shape)
        t=self.proj(t)
        #print(t.shape)
        return t


class ResidualDecompEncoder(nn.Module):
    def __init__(self,d_model,n_head,window_list=[2+1]):
        super(ResidualDecompEncoder,self).__init__()
        self.d_model=d_model
        
        self.count=len(window_list)
        self.attn_layers=nn.ModuleList([
            ResidualDecompEncoderLayer(d_model=d_model,n_head=n_head,window_size=size)
            for size in window_list
        ]) 

    def forward(self,x):
        x_=torch.zeros_like(x)
        #print(x_.shape)
        for layer in self.attn_layers:
            s=x
            x=layer(x)
            x_=x_+x
            x=s-x   #residual
        
        return x_
