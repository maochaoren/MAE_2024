import torch.nn as nn
import torch
import numpy as np
import math

class PositionalEmbedding(nn.Module):#PE
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print(self.pe[:,:x.size(1)].shape)
        return self.pe[:, :x.size(1)]

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, freq='h'):
        super(TemporalEmbedding, self).__init__()

        hour_size = 24; minute_size=4
        weekday_size = 7; day_size = 32; month_size = 13

        #Embed = nn.Embedding
        if freq=='t':
            self.minute_embed=nn.Embedding(minute_size,d_model)
        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)
    
    def forward(self, x):
        x = x.long()#x_mask的4个维度分别是month,day(月日号),day(一周中的第几天),hour
        minute_x=self.minute_embed(x[:,:,4]) if hasattr(self,'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return minute_x+ hour_x+ weekday_x + day_x + month_x


class Embedding(nn.Module):#若通道独立：共享参数
    def __init__(self,c_in,d_model,c_num,CI=False,pos_embed=True):
        super(Embedding, self).__init__()
        self.conv_embed=nn.Conv1d(in_channels=c_in,out_channels=d_model,kernel_size=3,padding=1,padding_mode='circular')
        self.positional_embed=PositionalEmbedding(d_model)
        self.CI=CI
        self.c_in=c_in
        self.c_num=c_num
        self.d_model=d_model
        self.pos_embed=pos_embed

    def forward(self,x):
        if self.CI:
            #x_=torch.zeros(x.shape[0],x.shape[1],self.d_model*self.c_num).to(x.device)
                #x_dim=x[:,:,self.c_in*dim:self.c_in*(dim+1)].clone()
            x=self.conv_embed(x.permute(0,2,1).float()).transpose(1,2)
            pos_embed=self.positional_embed(torch.zeros(x.shape))
            x=x+self.pos_embed if self.pos_embed else x
            #print(x.shape)
        else:
            x=self.conv_embed(x.permute(0,2,1).float()).transpose(1,2)
            pos_embed=self.positional_embed(torch.zeros(x.shape))
            x=x+pos_embed if self.pos_embed else x

        return x
       
       
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class Patching(nn.Module):
    def __init__(self, patch_len):
        super(Patching, self).__init__()
        # Patching
        self.patch_len = patch_len
        #self.stride = stride
        self.value_embedding = nn.Linear(in_features=patch_len,out_features=1,bias=True)

        # Positional embedding
        #self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x):
        # do patching
        _,L,_=x.shape
        x = torch.reshape(x, (x.shape[0] * (L//self.patch_len),self.patch_len, x.shape[2]))
        #print(x.shape)
        x = self.value_embedding(x.permute(0,2,1).float()).transpose(1,2)# x:(B*nvar*patch_num)*1*dmodel
        #print(x.shape)

        x = torch.reshape(x,((x.shape[0] // (L//self.patch_len)),(L//self.patch_len),x.shape[2]))#x:(B*nvar)*patch_num*d_model
        return x

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1).float()).transpose(1, 2)
        return x





