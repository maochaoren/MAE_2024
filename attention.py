import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat

class Mutihead_Attention(nn.Module):
    def __init__(self,d_model,dim_k,dim_v,n_heads=4,requires_mask=False):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads
        self.requires_mask=requires_mask
        self.d_model=d_model

        self.q = nn.Linear(d_model,dim_k)
        self.k = nn.Linear(d_model,dim_k)
        self.v = nn.Linear(d_model,dim_v)

        self.softmax=nn.Softmax(dim=2)
        #self.o = nn.Linear(dim_v,d_model)
        self.norm_fact = 1 / math.sqrt(d_model)
        self.attn=nn.MultiheadAttention(embed_dim=d_model,num_heads=n_heads,dropout=0.1)

    def forward(self,h1,h2,q_,k_,v_,mask=None,use_q=False):
        #assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]
        # 对 x 进行自注意力
        #print(h1.shape)
        if not use_q:
            Q=self.q(h1).float()
            K=self.k(h2).float()
            V=self.v(h2).float()
        else:
            Q=self.q(q_).float()
            #print(k_.shape,q_.shape)
            K=self.k(k_).float()
            V=self.v(v_).float()
        
        _,l_q,_=Q.shape
        _,l_v,_=V.shape
        if l_q>l_v:
            zeros=torch.zeros_like(Q[:, :(l_q - l_v), :]).float()
            V=torch.cat([V, zeros], dim=1)
            K=torch.cat([K, zeros], dim=1)
        else:
            V=V[:,:l_q,:]
            K=K[:,:l_q,:]
            #print(Q.shape,V.shape)
        #print(Q.view(Q.shape[0],-1,self.n_heads,self.dim_k // self.n_heads).shape)

        '''Q = Q.view(Q.shape[0],-1,self.n_heads,self.dim_k // self.n_heads).permute(0,2,1,3) # n_heads * batch_size * seq_len * dim_k
        #print(Q.shape)
        K = K.view(K.shape[0],-1,self.n_heads,self.dim_k // self.n_heads).permute(0,2,1,3) # n_heads * batch_size * seq_len * dim_k
        #print(K.shape)
        V = V.view(V.shape[0],-1,self.n_heads,self.dim_k // self.n_heads).permute(0,2,1,3) # n_heads * batch_size * seq_len * dim_k
        #print("Attention V shape : {}".format(V.shape))
        attention_score = (torch.matmul(Q,K.permute(0,1,3,2)) * self.norm_fact)
        #print(attention_score.shape)
        #print(attention_score.shape)
        #print(mask.shape)
        #print(mask.shape)
        if self.requires_mask:
            mask=mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
            attention_score=attention_score+mask # 注意这里的小Trick 不需要将Q,K,V 分别MASK,只MASKSoftmax之前的结果就好了
        attention_score=self.softmax(attention_score)
        #print(attention_score.shape)
        #print(V.shape)
        output = torch.matmul(attention_score,V).reshape(h2.shape[0],h2.shape[1],-1)'''

        output,_=self.attn(Q,K,V)
        return output
    
class Feed_Forward(nn.Module):
    def __init__(self,input_dim,hidden_dim=2048):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim,hidden_dim)
        self.L2 = nn.Linear(hidden_dim,input_dim)
        self.Dropout1=nn.Dropout(0.1)

    def forward(self,x):
        output = nn.ReLU()(self.L1(x))
        output = self.L2(output)
        return self.Dropout1(output)

    
class DualInputTransformerEncoder(nn.Module):
    def __init__(self,d_model,norm_first=True):
        super(DualInputTransformerEncoder, self).__init__()
        self.norm_first=norm_first
        #self.positional_encoding = Positional_Encoding(config.d_model)
        self.attn = Mutihead_Attention(d_model,d_model,d_model,n_heads=4,requires_mask=True)
        self.feed_forward = Feed_Forward(d_model,hidden_dim=2048)
        self.Dropout=nn.Dropout(0.1)
        self.LayerNorm1=nn.LayerNorm(d_model)
        self.LayerNorm2=nn.LayerNorm(d_model)
        #self.softmax=F.softmax(dim=-1)
        #self.add_norm = Add_Norm()
        #self.dropout=nn.Dropout(0.1)


    def forward(self,h1,h2,attn_mask): # batch_size * seq_len 并且 x 的类型不是tensor，是普通list

        #x += self.positional_encoding(x.shape[1],config.d_model)
        # print("After positional_encoding: {}".format(x.size()))
        if self.norm_first:
            #print(self.LayerNorm1(h1).shape)
            #print(h1.shape)
            x=h1+F.softmax((self.attn(self.LayerNorm1(h1),self.LayerNorm1(h2),attn_mask)),dim=-1)
            x=x+self.feed_forward(self.LayerNorm2(x))
        else:
            x=self.LayerNorm1(h1+F.softmax((self.attn(h1,h2,attn_mask)),dim=-1))
            x=self.LayerNorm2(x+self.feed_forward(x))
        
        return x
        
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
