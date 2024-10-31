import torch.nn as nn
import torch
from Embedding import PatchEmbedding,PositionalEmbedding
from RetroMAElike_model import instance_denorm,instance_norm,DilatedConvEncoder,RevIN
from MultiDecompMAE import series_decomp,MultiDecompEncoder
from SimMTM import SimMTM,fft_decomp,topk_fft_decomp
from attention import Mutihead_Attention,Feed_Forward,AttentionLayer,FullAttention
from Transformer_Layers import MultiDecompTransformerEncoderLayer,ResidualDecompEncoder,Encoder,EncoderLayer
from Non_Stationary_Transformer import ns_transformer
import torch.nn.functional as F

class DecoderLayer(nn.Module):

    def __init__(self,c_in, d_model,window_size=25):
        super(DecoderLayer,self).__init__()
        self.d_model=d_model
        self.self_attn=Mutihead_Attention(d_model=d_model,dim_k=d_model,dim_v=d_model,requires_mask=False,n_heads=4)
        self.cross_attn=Mutihead_Attention(d_model=d_model,dim_k=d_model,dim_v=d_model,requires_mask=False,n_heads=4)
        self.series_decomp1=series_decomp(kernel_size=window_size)
        self.series_decomp2=series_decomp(kernel_size=window_size)
        self.series_decomp3=series_decomp(kernel_size=window_size)
        self.feed_forward=Feed_Forward(input_dim=d_model)
        self.Dropout=nn.Dropout(0.1)
        self.layernorm=nn.LayerNorm(d_model)
        self.conv1=nn.Conv1d(in_channels=d_model,out_channels=d_model*4,kernel_size=3,bias=False,padding=1,padding_mode='circular')
        self.conv2=nn.Conv1d(in_channels=d_model*4,out_channels=d_model,kernel_size=3,bias=False,padding=1,padding_mode='circular')
        self.activation=F.relu
        self.proj_t=nn.Conv1d(in_channels=c_in,out_channels=c_in,kernel_size=3,bias=False,padding=1,padding_mode='circular')
        self.t1=nn.Linear(d_model,c_in)
        self.t2=nn.Linear(d_model,c_in)
        self.t3=nn.Linear(d_model,c_in)
        self.t4=nn.Linear(c_in,c_in)

    def forward(self,x,cross,trend=None):
        x=self.layernorm(x)
        x=x+self.Dropout(self.self_attn(x,x,x,x,x))
        x,trend1=self.series_decomp1(x)

        x=x+self.Dropout(self.cross_attn(x,x,x,cross,cross,use_q=True))
        x,trend2=self.series_decomp2(x)

        #feed_forward
        y=x
        y=self.Dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y=self.Dropout(self.conv2(y).transpose(-1,1))
        
        x,trend3=self.series_decomp3(x+y)

        trend=self.t1(trend1)+self.t2(trend2)+self.t3(trend3)+(self.t4(trend.float()))

        #trend_res=self.proj_t(trend.permute(0,2,1).float()).transpose(1,2)

        #trend=self.t4(trend)+trend_res

        return x,trend

class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x

        
class FineTuningModelNew(nn.Module):
    def __init__(self,c_in,d_model,n_head,encoder_depth,input_len,series_embed_len,mae_encoder,pred_len,window_size=24+1,dom_season=24,use_cls_token=False,lpf=0,
                 is_norm=False,is_decomp=False,frozen_num=3,use_decoder=False,random_init=False,CI=False,backbone='res',part='s',t_model='transformer',decomp='fft',st_sep=0.5,topk=15
        ):
        super(FineTuningModelNew, self).__init__()
        self.positional_embed=PositionalEmbedding(d_model=d_model)
        self.input_len=input_len
        self.encoder_depth=encoder_depth
        self.pred_len=pred_len
        self.c_in=c_in
        self.d_model=d_model
        self.is_norm=is_norm
        self.CI=CI
        self.frozen_num=frozen_num
        self.random_init=random_init

        self.series_decomp=series_decomp(kernel_size=window_size)
        self.fft_decomp=fft_decomp(st_sep=st_sep,padding_rate=9,lpf=lpf)
        self.topk_fft_decomp=topk_fft_decomp(k=topk)
        
        self.part=part
        self.t_model=t_model
        self.decomp=decomp
        #encoder
        self.enc_embedding=nn.Conv1d(in_channels=1 if CI else c_in,out_channels=d_model,kernel_size=3,padding=1,padding_mode='circular')

        self.random_init_encoder=SimMTM(mask_rate=0.3,window_size=window_size,encoder_depth=encoder_depth,st_sep=st_sep,c_in=c_in,d_model=d_model,n_head=n_head,is_norm=True,time_block=1,window_list=[24+1,12+1,1],CI=CI,
                                                    input_len=input_len,is_decomp=is_decomp,backbone=backbone,topk=topk)
        
        self.norm_linear=nn.Linear(1 if CI else c_in,1 if CI else c_in)
        self.encoder=self.random_init_encoder if random_init else mae_encoder 
        #self.dilated_conv=DilatedConvEncoder(1,[d_model]*10,kernel_size=3)
        #self.ns_transformer=ns_transformer(d_model=d_model,c_in=c_in,CI=CI,input_len=input_len,pred_len=pred_len,encoder_depth=2)

        #decoder
        self.proj_dec=nn.Conv1d(c_in,d_model,kernel_size=3,padding=1,padding_mode='circular')
        self.decoder_proj=nn.Linear(d_model,c_in,bias=True)
        
        self.decoder_pred=nn.Linear(input_len+series_embed_len,pred_len) if use_cls_token else nn.Linear(input_len,pred_len)
        self.pred_s=nn.Linear(input_len,pred_len)
        self.pred_t=nn.Linear(input_len,pred_len)

        self.proj_s=nn.Linear(in_features=d_model,out_features=1 if CI else c_in)
        #self.proj_s=nn.Conv1d(d_model,1,kernel_size=3,stride=1,padding=1,padding_mode='circular')
        self.proj_t=nn.Linear(in_features=d_model,out_features=1 if CI else c_in)

        self.head_s=Flatten_Head(seq_len=input_len, d_model=d_model, pred_len=pred_len)
        self.head_t=Flatten_Head(seq_len=input_len, d_model=d_model if self.t_model=='transformer' else 1, pred_len=pred_len)

        self.is_decomp=is_decomp

        #Autoformer-like decoder
        self.use_decoder=use_decoder
        self.decoder=DecoderLayer(d_model=d_model,c_in=c_in,window_size=window_size)

    def forward_pred(self,x):#只使用线性层预测，不再使用transformer decoder
        pred=self.decoder_proj(x)
        pred=self.decoder_pred(pred.permute(0,2,1).float()).transpose(1,2)

        return pred

    
    def forward_loss(self,pred,pred_label):
        x=pred
        pred_label=pred_label.float()
        loss_func=nn.MSELoss()
        loss = loss_func(x,pred_label).float()
        return loss
    
    def dim_loss(self,pred,pred_label,dim):
        x=pred[:,:,dim]
        pred_label=pred_label[:,:,dim].float()
        loss_func=nn.MSELoss()
        loss = loss_func(x,pred_label).float()
        return loss

    def forward(self,x,x_pred):

        if self.is_decomp:
            '''x_s,x_t=self.series_decomp(x)
            if self.is_norm:
                x_t,means,stdev=instance_norm(x_t)'''

            if not self.use_decoder:
                
                if self.is_norm:
                    #x_t,means,stdev=instance_norm(x_t)
                    x,means,stdev=instance_norm(x)
                    #x_=self.norm_linear(x_t_raw)

                if self.decomp=='fft':
                    x_s,x_t=self.fft_decomp(x)
                    x_s,x_res=self.topk_fft_decomp(x_s)
                elif self.decomp=='mov_avg':
                    x_s,x_t=self.series_decomp(x)
                elif self.decomp=='topk_fft':
                    x_s,x_t=self.topk_fft_decomp(x)

                stdev = torch.sqrt(
                    torch.var(x_t, dim=1, keepdim=True, unbiased=False) + 1e-5)
                x_t=x_t/stdev
                
                if self.CI:
                    x_s=x_s.permute(0,2,1)
                    x_t=x_t.permute(0,2,1)
                    #print(x_s.shape)
                    x_s=torch.reshape(x_s,(x_s.shape[0]*x_s.shape[1],x_s.shape[2])).unsqueeze(-1)
                    x_t=torch.reshape(x_t,(x_t.shape[0]*x_t.shape[1],x_t.shape[2])).unsqueeze(-1)
                
                #x_s,x_t=self.encoder.repr_gen(x_s,x_t,frozen_num=self.frozen_num)
                if self.t_model=='transformer':
                    x_s,x_t=self.encoder.repr_gen(x_s,x_t,frozen_num=0 if self.random_init else self.frozen_num,is_ft=False if self.random_init else True)
                else:
                    x_s,_=self.encoder.repr_gen(x_s,x_t,frozen_num=0 if self.random_init else self.frozen_num,is_ft=False if self.random_init else True)
                #x_t=self.ns_transformer(x_t)
                #print(x_s.shape)
                    
                '''x_s=torch.reshape(x_s,(x_s.shape[0]//self.c_in,self.c_in,x_s.shape[1],x_s.shape[2]))
                x_t=torch.reshape(x_t,(x_t.shape[0]//self.c_in,self.c_in,x_t.shape[1],x_t.shape[2]))
                x_s=self.head_s(x_s).permute(0,2,1)
                x_t=self.head_t(x_t).permute(0,2,1)'''
                
                x_s=self.pred_s(x_s.permute(0,2,1).float()).transpose(1,2)
                x_s=self.proj_s(x_s)
                
                x_t=self.pred_t(x_t.permute(0,2,1).float()).transpose(1,2)
                #print(x_t.shape)
                x_t=self.proj_t(x_t) if self.t_model=='transformer' else x_t

                    #x_t=x_t+x_
                #reshape
                if self.CI:
                    x_s=torch.reshape(x_s,(x_s.shape[0]//self.c_in,self.c_in,x_s.shape[1]))
                    x_t=torch.reshape(x_t,(x_t.shape[0]//self.c_in,self.c_in,x_t.shape[1]))
                    x_s=x_s.permute(0,2,1)
                    x_t=x_t.permute(0,2,1)
 
                #x_t=instance_denorm(x_t,means,stdev,self.pred_len)
                
                x_t = x_t * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                pred=x_s+x_t

                if self.is_norm:
                    pred=instance_denorm(pred,means,stdev,self.pred_len)

                
                '''loss_s=self.forward_loss(pred=x_s,pred_label=x_s_)
                loss_t=self.forward_loss(pred=x_t,pred_label=x_t_)'''
                loss_s=0
                loss_t=0
                #loss_total=self.forward_loss(pred=pred,pred_label=x_pred)
                loss_total=self.forward_loss(pred=pred,pred_label=x_pred)    

                '''if self.part=='s':
                    loss=loss_s
                elif self.part=='t':
                    loss=loss_t
                else:
                    loss=loss_total'''
                loss=loss_total

                '''if self.part=='s':
                    return loss,x_s
                elif self.part=='t':
                    return loss,x_t
                else:
                    return loss,pred'''

                return loss,loss_s,loss_t,pred
                            
            elif self.use_decoder:

                '''if self.is_norm:
                    x_norm,means,stdev=instance_norm(x)
                x_s,x_t=self.series_decomp(x_norm)'''

                x_s,x_t=self.series_decomp(x)
                if self.is_norm:
                    x_t,means,stdev=instance_norm(x_t)

                zeros = torch.zeros([x.shape[0], self.pred_len,
                             x.shape[2]], device=x.device)
                t_init=torch.cat([x_t,zeros],dim=1)
                s_init=torch.cat([x_s,zeros],dim=1)

                enc_out,_=self.encoder.repr_gen(x_s,x_t,frozen_num=self.frozen_num)
                #enc_out,_=self.encoder_.repr_gen(x_s,x_t,frozen_num=self.frozen_num)
                #dec
                #embed
                #print(s_init.shape)
                dec_out=self.proj_dec(x_s.permute(0,2,1).float()).transpose(1,2)
                #dec_out+=self.positional_embed(dec_out)

                x_s,x_t=self.decoder(dec_out,enc_out,x_t)
                #print(x_s.shape,x_t.shape)
                x_s=self.proj_s(x_s)

                #pred=x_t+x_s
                #print(x_t.shape,x_s.shape)

                x_s=self.pred_s(x_s.permute(0,2,1).float()).transpose(1,2)
                x_t=self.pred_t(x_t.permute(0,2,1).float()).transpose(1,2)

                if self.is_norm:
                    x_t=instance_denorm(x_t,means,stdev,self.pred_len)

                pred=x_s+x_t
                #pred=pred[:,self.input_len:,:]

                '''if self.is_norm:
                    pred=instance_denorm(pred,means,stdev,self.pred_len)'''
                #print(pred.shape)

                loss=self.forward_loss(pred=pred,pred_label=x_pred)

                
        else:
            if self.is_norm:
                x,means,stdev=instance_norm(x)
                means=self.mean_linear(means)
                stdev=self.stdev_linear(stdev)


            #print(stdev.shape)
            if self.CI:
                x=x.permute(0,2,1)
                #print(x_s.shape)
                x=torch.reshape(x,(x.shape[0]*x.shape[1],x.shape[2])).unsqueeze(-1)
            x,_=self.encoder.repr_gen(x,x,frozen_num=0 if self.random_init else self.frozen_num)

            x=self.proj_s(x)

            pred=self.pred_s(x.permute(0,2,1).float()).transpose(1,2)
            
            if self.CI:
                pred=torch.reshape(pred,(pred.shape[0]//self.c_in,self.c_in,pred.shape[1]))
                pred=pred.permute(0,2,1)

            if self.is_norm:
                pred=instance_denorm(pred,means,stdev,self.pred_len)

            loss=self.forward_loss(pred=pred,pred_label=x_pred)
            #print(loss)

            return loss,pred

