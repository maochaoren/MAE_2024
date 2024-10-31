import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from SimMTM import fft_decomp ,topk_fft_decomp
from Embedding import PatchEmbedding,PositionalEmbedding,Embedding
from RetroMAElike_model import instance_denorm,instance_norm
from Transformer_Layers import MultiDecompTransformerEncoderLayer,ResidualDecompEncoder,Encoder,EncoderLayer
from attention import AttentionLayer,FullAttention

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

class MultiDecompEncoder(nn.Module):
    def __init__(self,mask_rate,window_size,encoder_depth,c_in,d_model,n_head,input_len,is_norm,time_block,window_list=[1],CI=False,
                 is_decomp=True,backbone='res',part='s',st_sep=0.8,topk=50):
        super(MultiDecompEncoder, self).__init__()
        self.d_model=d_model
        self.input_len=input_len
        self.encoder_depth=encoder_depth
        #self.embed_dim=embed_dim
        self.c_in=c_in
        self.mask_rate=mask_rate
        self.mask_token=nn.Parameter(torch.zeros(1,1,d_model))
        self.layernorm=nn.LayerNorm(d_model)
        self.time_block=time_block
        self.flag=0
        self.CI=CI
        self.is_decomp=is_decomp
        self.part=part
        self.topk=topk

        self.series_decomp=series_decomp(kernel_size=window_size)
        self.fft_decomp=fft_decomp(st_sep=st_sep,padding_rate=9,lpf=0)
        self.topk_fft_decomp=topk_fft_decomp(k=15)

        self.embedding_enc_s=Embedding(c_in=1 if CI else c_in,d_model=d_model,c_num=c_in,CI=CI,pos_embed=True)
        self.embedding_enc_t=Embedding(c_in=1 if CI else c_in,d_model=d_model,c_num=c_in,CI=CI,pos_embed=False)

        '''self.encoder_s=nn.ModuleList([
            MultiDecompTransformerEncoderLayer(c_in=c_in,d_model=d_model,input_len=input_len,window_size=24+1)
            for i in range(0,encoder_depth)
        ])'''
        self.transformer_encoder=Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False,
                                      output_attention=False), d_model, n_heads=n_head),
                    d_model
                ) for l in range(encoder_depth)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.res_encoder=ResidualDecompEncoder(d_model=d_model,n_head=n_head,window_list=window_list)

        self.encoder_s=self.transformer_encoder if backbone=='vanilla' else self.res_encoder

        self.transformer_encoder_t=Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False,
                                      output_attention=False), d_model, n_heads=n_head),
                    d_model
                ) for l in range(encoder_depth)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        #self.encoder=MultiDecompTransformerEncoderLayer(c_in=c_in,d_model=d_model,input_len=input_len)

        self.dec_embed_s=Embedding(c_in=d_model,d_model=d_model,c_num=c_in,CI=CI,pos_embed=True)
        self.dec_embed_t=Embedding(c_in=d_model,d_model=d_model,c_num=c_in,CI=CI,pos_embed=False)

        self.dec_conv_s=nn.Conv1d(d_model,d_model,kernel_size=3,stride=1,padding=1,padding_mode='circular')
        self.dec_conv_t=nn.Conv1d(d_model,d_model,kernel_size=3,stride=1,padding=1,padding_mode='circular')

        self.proj_s=nn.Linear(in_features=d_model,out_features=1 if CI else c_in)
        self.proj_t=nn.Linear(in_features=d_model,out_features=1 if CI else c_in)
        self.is_norm=is_norm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def mask_func(self,x_s,mask_ratio):
        
        N, _, D = x_s.shape # batch, length(num_patches), dim
        L=int(self.input_len)
        len_keep = int(L * (1 - mask_ratio))

        mask_rand=torch.rand(N,L,device=x_s.device)
        ids_shuffle = torch.argsort(mask_rand, dim=1)  # ascend: small is keep, large is remove(和mask_ratio相比) 其实是第0维（同一行）之间排序
        ids_keep_restore = torch.sort(ids_shuffle[:,:len_keep], dim=1).values
        ids_restore=torch.argsort(torch.cat([ids_keep_restore,ids_shuffle[:,len_keep:]],dim=1),dim=1)#不改变保留token的相对次序
        #小的留下，大的mask掉
        #print(x.shape)
        x_s_masked = torch.gather(x_s, dim=1, index=ids_keep_restore.unsqueeze(-1).repeat(1, 1, D).long())#gather:按照给定的idx挑选第dim维保留的元素，然后将第dim维消去。
        #print(x_masked.shape)
        mask = torch.ones([N, L], device=x_s.device)
        mask[:, :len_keep] = 0 #在mask数组中，0表示未被Mask，1表示被Mask的
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_s_masked, mask, ids_restore.long()
    
    def mask_func_block(self,x,mask_ratio,is_rand):#按块为单位进行mask，随机或固定位置mask
        N, _, D = x.shape # batch, length(num_patches), dim
        num_block=int(self.input_len/self.time_block) #time_block:时间块长度
        L=int(self.input_len)

        if not is_rand:#若mask_rate=75%，则每1个非mask后接3个mask token。
            patch_num=int(1+mask_ratio/(1-mask_ratio))
            mask=torch.ones([N,L],device=x.device)
            #print(self.time_block)
            for i in range(0,num_block):
                if i % patch_num==0:#保留
                    mask[:,i*self.time_block:(i+1)*self.time_block]=1
                else:#被mask
                    mask[:,i*self.time_block:(i+1)*self.time_block]=0

            mask=mask.unsqueeze(-1).repeat(1,1,self.d_model)
            x_masked=x*mask
            mask=~(mask.bool())
            mask=mask[:,:,0].int()

            ids_restore=torch.zeros((N,num_block),device=x.device)
            for i in range(0,int(num_block/patch_num)):
                ids_restore[:,i*patch_num]=i
                for j in range(1,patch_num):
                    ids_restore[:,i*patch_num+j]=num_block/patch_num+i*(patch_num-1)+j-1

            #增广ids_restore
            ids_restore_expand=torch.zeros((N,self.input_len),device=x.device)
            for i in range(0,num_block):
                for j in range(0,self.time_block):
                    ids_restore_expand[:,i*self.time_block+j]=ids_restore[:,i]*self.time_block+j

            return x_masked,mask,ids_restore_expand.long()

        elif is_rand:

            mask=torch.ones([N,num_block],device=x.device)
            len_keep=int(num_block*(1-mask_ratio))

            mask_rand=torch.rand(N,num_block,device=x.device)
            ids_shuffle = torch.argsort(mask_rand, dim=1)  # ascend: small is keep, large is remove(和mask_ratio相比) 其实是第0维（同一行）之间排序
            ids_keep_restore = torch.sort(ids_shuffle[:,:len_keep], dim=1).values
            ids_restore=torch.argsort(torch.cat([ids_keep_restore,ids_shuffle[:,len_keep:]],dim=1),dim=1)
            #小的留下，大的mask掉
            #print(x_masked.shape)
            mask[:, :len_keep] = 0 #在mask数组中，0表示保留，1表示被Mask的
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            #增广
            mask_expand=torch.zeros((N,self.input_len),device=x.device)
            ids_keep_restore_expand = torch.zeros((N,len_keep*self.time_block),device=x.device)
            ids_restore_expand=torch.zeros((N,self.input_len),device=x.device)
            for i in range(0,num_block):
                for j in range(0,self.time_block):
                    ids_restore_expand[:,i*self.time_block+j]=ids_restore[:,i]*self.time_block+j

            for i in range(0,num_block):
                mask_expand[:,i*self.time_block:(i+1)*self.time_block]=mask[:,i].unsqueeze(1)

            for i in range(0,len_keep):
                for j in range(0,self.time_block):
                    ids_keep_restore_expand[:,i*self.time_block+j]=ids_keep_restore[:,i]*self.time_block+j

            x_masked = torch.gather(x, dim=1, index=ids_keep_restore_expand.unsqueeze(-1).repeat(1, 1, D).long())#gather:按照给定的idx挑选第dim维保留的元素，然后将第dim维消去。

            return x_masked,mask_expand,ids_restore_expand.long()

    def forward_loss(self,pred_label,pred,mask):
        x=pred
        #计算MSE Loss
        loss = (x-pred_label) ** 2
        #print(x)
        loss = loss.mean(dim=-1) #loss:B x (L x P)
        #print(loss*mask)
        loss=(loss*mask).sum()/mask.sum()
        
        return loss
    
    def dim_loss(self,pred_label,pred,dim):#返回每个维度的损失
        x=pred[:,:,dim]
        #print(x)
        pred_label=pred_label[:,:,dim]

        loss = (x-pred_label) ** 2
        #print(loss)
        #print(loss*self.mask)
        loss=(loss*self.mask).sum()/self.mask.sum()
        return loss
    
    def forward(self,x):
        #decomp & norm
        #x_s_,x_t_=self.series_decomp(x)#x_s_,x_t_分别作为两部分的ground truth
        x,means,stdev=instance_norm(x)
        if self.decomp=='fft':
            x_s_,x_t_=self.fft_decomp(x)
            x_s_,x_res=self.topk_fft_decomp(x_s_)
        elif self.decomp=='mov_avg':
            x_s_,x_t_=self.series_decomp(x)
        
        x_t=x_t_.clone()
        stdev = torch.sqrt(
            torch.var(x_t, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_t=x_t/stdev
        x_s=self.topk_fft_decomp(x_s_)[0] if self.decomp=='fft' else x_s_.clone()

        if self.CI:
            x_s_=x_s_.permute(0,2,1)
            x_s_=torch.reshape(x_s_,(x_s_.shape[0]*x_s_.shape[1],x_s_.shape[2])).unsqueeze(-1)
            x_t_=x_t_.permute(0,2,1)
            x_t_=torch.reshape(x_t_,(x_t_.shape[0]*x_t_.shape[1],x_t_.shape[2])).unsqueeze(-1)

        '''if self.is_norm:
            if not self.is_decomp:
                x_norm,means,stdev=instance_norm(x)
                x_s=x_norm
            else:
                x_t,means,stdev=instance_norm(x_t_)
                x_s=x_s_ if self.part=='s' else x_t
        else:
            x_s=x'''
        

        #reshape -> CI               
        #mask
        x_s_masked,mask_s,ids_restore_s=self.mask_func_block(x_s,self.mask_rate,is_rand=True)#mask
        #x_t_masked,mask_t,ids_restore_t=self.mask_func(x_t,self.mask_rate)
        #embed
        #print(x_s_masked.shape)
        x_s_masked=self.embedding_enc_s(x_s_masked)
        #分解序列 init
        #print(x_masked.shape)
        x_s_masked=self.encoder_s(x_s_masked)
        #x_s_masked,_=self.encoder_s(x_s_masked)

        #x_s padding
        mask_tokens_s=nn.Parameter(torch.zeros(ids_restore_s.shape[0],int(self.input_len*self.mask_rate),self.d_model)).to(x.device)
        #print(x_s_masked.shape)
        x_s=torch.cat([x_s_masked,mask_tokens_s],dim=1)
        x_s=torch.gather(x_s,dim=1,index=ids_restore_s.unsqueeze(-1).repeat(1, 1, x_s.shape[2]))#复原被打乱的x_mask序列
            

        #decoding
        #dec_embed
        x_s=self.dec_conv_s(x_s.permute(0,2,1).float()).transpose(1,2)
        x_s=self.proj_s(x_s)

        if self.is_norm and((not self.is_decomp) or self.part=='t'):
            x_s=instance_denorm(x_s,means,stdev,self.input_len)
        
        if self.is_decomp:
            loss_s=self.forward_loss(pred_label=x_s_,pred=x_s,mask=mask_s) if self.part=='s' else self.forward_loss(pred_label=x_t_,pred=x_s,mask=mask_s)
        else:
            loss_s=self.forward_loss(pred_label=x,pred=x_s,mask=mask_s)
        #loss=(loss_s+loss_t)/2
        loss=loss_s

        #print(x_s.shape)
        if self.CI:
            x_s=torch.reshape(x_s,(x_s.shape[0]//self.c_in,x_s.shape[1],self.c_in))
        #print(x_s.shape)
        #pred=x_t
        '''pred=x_s+x_t
        #拼接两部分而不是相加
        x_=torch.concat([x_s,x_t],dim=2)
        pred=self.proj_s(x_)
        if self.is_norm:
            pred=instance_denorm(pred,means,stdev,self.input_len)

        loss_func=nn.MSELoss()
        loss=loss_func(x,pred)'''

        #print(loss)
        return x_s,loss   

    def repr_gen(self,x_s,x_t,frozen_num,is_ft=True):



        x_s=self.embedding_enc_s(x_s)
        x_t=self.embedding_enc_t(x_t)

        x_s=self.encoder_s(x_s)
            
        x_t=self.encoder_t(x_t)
        #x_t=self.encoder_t(x_t)

        #x_t=self.encoder_t(x_t)

        return x_s,x_t


    






    





