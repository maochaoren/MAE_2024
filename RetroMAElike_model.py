import torch.nn as nn
import torch.nn.functional as F
import torch
from Embedding import PatchEmbedding,PositionalEmbedding
from attention import DualInputTransformerEncoder 

def instance_norm(x):
    #instance_norm
    #print(x)
    means = torch.mean(x,dim=1,keepdim=True).detach()
    #means = means.unsqueeze(1)
    #print(means.shape)
    x = x - means
    #print(x)
    #x = x.masked_fill(0)
    stdev = torch.sqrt(
        torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    #x =x/ stdev
    return x,means,stdev

def instance_denorm(x,means,stdev,seq_len):
        #x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        return x

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x



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


class RetroMAE_Encoder(nn.Module):
    def __init__(self,c_in,d_model,input_len,series_embed_len,mask_rate_enc,encoder_depth,mask_size,train_pe=True,is_norm=False):
        super(RetroMAE_Encoder,self).__init__()
        self.is_norm=is_norm
        self.d_model=d_model
        self.input_len=input_len
        self.c_in=c_in
        self.series_embed_len=series_embed_len
        self.mask_rate_enc=mask_rate_enc
        self.mask_token=nn.Parameter(torch.zeros(1,1,d_model))
        self.time_block=24
        #encoder
        self.train_pe=train_pe
        self.ScalarProjection_enc=nn.Conv1d(c_in,d_model,kernel_size=3,padding=1,padding_mode='circular')
        self.PositionEmbedding=PositionalEmbedding(d_model=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1,self.series_embed_len,d_model))
        Transformer_Encoder_Layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=4,norm_first=True,batch_first=False)
        self.Transformer_Encoder=nn.TransformerEncoder(Transformer_Encoder_Layer,num_layers=encoder_depth)
        self.encoder_proj=nn.Linear(d_model,c_in,bias=True)
        self.encoder_pred=nn.Linear(input_len+series_embed_len,input_len,bias=True)
        
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
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def mask_func(self,x,mask_ratio):
        
        N, _, D = x.shape # batch, length(num_patches), dim
        L=int(self.input_len)
        len_keep = int(L * (1 - mask_ratio))

        mask_rand=torch.rand(N,L,device=x.device)
        ids_shuffle = torch.argsort(mask_rand, dim=1)  # ascend: small is keep, large is remove(和mask_ratio相比) 其实是第0维（同一行）之间排序
        ids_keep_restore = torch.sort(ids_shuffle[:,:len_keep], dim=1).values
        ids_restore=torch.argsort(torch.cat([ids_keep_restore,ids_shuffle[:,len_keep:]],dim=1),dim=1)
        #小的留下，大的mask掉
        x_masked = torch.gather(x, dim=1, index=ids_keep_restore.unsqueeze(-1).repeat(1, 1, D).long())#gather:按照给定的idx挑选第dim维保留的元素，然后将第dim维消去。
        #print(x_masked.shape)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0 #在mask数组中，0表示保留，1表示被Mask的
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore.long()
    
    def mask_func_block(self,x,mask_ratio,is_rand):#按块为单位进行mask，随机或固定位置mask
        N, _, D = x.shape # batch, length(num_patches), dim
        num_block=int(self.input_len/self.time_block) #time_block:时间块长度
        L=int(self.input_len)

        if not is_rand:#若mask_rate=75%，则每1个非mask后接3个mask token。
            patch_num=1+mask_ratio/(1-mask_ratio)
            mask=torch.ones([N,L],device=x.device)

            for i in range(0,num_block):
                if i % patch_num==0:#保留
                    mask[:,i:i+self.time_block]=1
                else:#被mask
                    mask[:,i:i+self.time_block]=0

            mask=mask.repeat(1,1,self.d_model)
            x_masked=x*mask
            mask=~mask

            ids_restore=torch.zeros((N,num_block))
            for i in range(0,num_block/patch_num):
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

    def forward_loss(self,x_label,pred,mask):
        x=pred
        #计算MSE Loss
        loss = (x-x_label) ** 2
        #print(x)
        loss = loss.mean(dim=-1) #loss:B x (L x P)
        loss=(loss*mask).sum()/mask.sum()
        #print(loss.shape)
        return loss
    
    def dim_loss(self,pred_label,pred,dim):#返回每个维度的损失
        x=pred[:,:,dim]
        #print(x)
        pred_label=pred_label[:,:,dim]

        loss = (x-pred_label) ** 2

        loss=(loss*self.mask).sum()/self.mask.sum()
        return loss
    
    def repr_gen(self,x,pe,fine_tune=False):#生成表征
        pos_embed=self.PositionEmbedding(torch.zeros(x.shape[0],self.input_len+self.series_embed_len))
        if fine_tune:
            if(pe):
                x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
            else:
                x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
            
            #添加cls_token，cls_token生成的embedding作为整个时间序列的embedding
            cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
            cls_tokens = cls_tokens+pos_embed[:,:self.series_embed_len]
            
            x=torch.cat([cls_tokens,x],dim=1)
    
            x=self.Transformer_Encoder(x.transpose(0,1)).transpose(0,1)
        else:
            with torch.no_grad():
                if(pe):
                    x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
                else:
                    x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)

                #添加cls_token，cls_token生成的embedding作为整个时间序列的embedding
                cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
                cls_tokens = cls_tokens+pos_embed[:,:self.series_embed_len]

                x=torch.cat([cls_tokens,x],dim=1)

                x=self.Transformer_Encoder(x.transpose(0,1)).transpose(0,1)

        return x
        
    def forward(self,x):
        x_label=x
        #normalization
        if self.is_norm:
            x,means,stdev=instance_norm(x)
        #Conv1d映射、添加pos_embed
        pos_embed=self.PositionEmbedding(torch.zeros(x.shape[0],self.input_len+self.series_embed_len))

        #embedding
        if(self.train_pe):
            x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
        else:
            x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
        #mask
        x_masked, mask, ids_restore=self.mask_func(x,self.mask_rate_enc)#mask
        self.mask=mask
        #添加cls_token，cls_token生成的embedding作为整个时间序列的embedding
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        cls_tokens = cls_tokens+pos_embed[:,:self.series_embed_len] if self.train_pe else cls_tokens
        
        x=torch.cat([cls_tokens,x_masked],dim=1)

        #输入encoder
        x=self.Transformer_Encoder(x.transpose(0,1)).transpose(0,1)
        series_embed=x[:,:self.series_embed_len,:]#时间序列embedding

        mask_tokens=nn.Parameter(torch.zeros((ids_restore.shape[0],int(self.input_len*self.mask_rate_enc)+1,x.shape[2]))).to(x.device)
        x=torch.cat([x[:,self.series_embed_len:,:],mask_tokens],dim=1)

        x=torch.gather(x,dim=1,index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))#复原被打乱的x_mask序列
        x=torch.cat([series_embed,x],dim=1)#加上cls_token
        
        #encoder阶段预测 按照ids_restore恢复，拼接0，通过linear预测？
        #embedding
        x=self.encoder_proj(x)
        x=self.encoder_pred(x.permute(0,2,1).to(torch.float32)).transpose(1,2)

        #denormalization
        if self.is_norm:
            x=instance_denorm(x,means,stdev,self.input_len)

        loss_enc=self.forward_loss(x_label,x,mask)
        return series_embed,loss_enc,x

class RetroMAE_Decoder(nn.Module):
    def __init__(self,c_in,d_model,input_len,series_embed_len,mask_rate_dec,mask_size,enhance_decoding=False,train_pe=True,is_norm=False):
        super(RetroMAE_Decoder,self).__init__()
        self.d_model=d_model
        self.is_norm=is_norm
        self.input_len=input_len
        self.c_in=c_in
        self.series_embed_len=series_embed_len
        self.mask_rate_dec=mask_rate_dec
        self.mask_token=nn.Parameter(torch.zeros(1,1,d_model))
        #decoder
        self.enhance_decoding=enhance_decoding
        self.train_pe=train_pe
        self.ScalarProjection_dec=nn.Conv1d(c_in,d_model,kernel_size=3,padding=1,padding_mode='circular')
        self.PositionEmbedding=PositionalEmbedding(d_model=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1,self.series_embed_len,d_model))
        Transformer_Decoder_Layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=4,norm_first=True,batch_first=False)
        self.Transformer_Decoder=nn.TransformerEncoder(Transformer_Decoder_Layer,num_layers=1)
        self.ScalarProjection_pred_dec=nn.Conv1d(d_model,d_model,kernel_size=3,padding=1,padding_mode='circular')
        self.decoder_pred=nn.Linear(input_len+series_embed_len,input_len,bias=True)
        self.decoder_proj=nn.Linear(d_model,c_in,bias=True)
       
        #enhanced decoder
        self.EnhancedDecoderLayer=DualInputTransformerEncoder(d_model=d_model,norm_first=True)
        self.criterion=nn.MSELoss()

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
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def mask_func(self,x,mask_ratio):
        N, _, D = x.shape # batch, length(num_patches), dim
        L=int(self.input_len)
        len_keep = int(L * (1 - mask_ratio))

        mask_rand=torch.rand(N,L,device=x.device)
        ids_shuffle = torch.argsort(mask_rand, dim=1)  # ascend: small is keep, large is remove(和mask_ratio相比) 其实是第0维（同一行）之间排序
        ids_keep_restore = torch.sort(ids_shuffle[:,:len_keep], dim=1).values
        ids_restore=torch.argsort(torch.cat([ids_keep_restore,ids_shuffle[:,len_keep:]],dim=1),dim=1)
        #小的留下，大的mask掉
        x_masked = torch.gather(x, dim=1, index=ids_keep_restore.unsqueeze(-1).repeat(1, 1, D).long())#gather:按照给定的idx挑选第dim维保留的元素，然后将第dim维消去。
        #print(x_masked.shape)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0 #在mask数组中，0表示未被Mask，1表示被Mask的
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore.long()
    
    def attn_mask(self,x,mask_ratio):
        N, _, D = x.shape # batch, length(num_patches), dim
        L=int(self.input_len+self.series_embed_len)
        len_keep=int(L*L*(1-mask_ratio))

        mask_rand=torch.rand(N,L**2,device=x.device)
        ids_shuffle=torch.argsort(mask_rand,dim=1)
        ids_restore=torch.argsort(ids_shuffle,dim=1) 

        mask=torch.ones(1,1,device=x.device)
        mask[0,0]=float("-inf")
        mask=mask.repeat(N,L**2)
        mask[:,:len_keep]=0

        attn_mask=torch.gather(mask,dim=1,index=ids_restore).reshape(N,L,L)

        #对角线元素屏蔽
        for i in range(0,L):
            attn_mask[:,i,i]=float("-inf")

        return attn_mask
    
    def forward_loss(self,x_label,pred,mask):
        x=pred
        #计算MSE Loss
        loss = (x-x_label) ** 2
        #print(x)
        loss = loss.mean(dim=-1) #loss:B x (L x P)
        loss=(loss*mask).sum()/mask.sum()
        #print(loss.shape)
        return loss
    
    def dim_loss(self,pred_label,pred,dim):#返回每个维度的损失
        x=pred[:,:,dim]
        #print(x)
        pred_label=pred_label[:,:,dim]

        loss = (x-pred_label) ** 2

        loss=(loss*self.mask).sum()/self.mask.sum()
        return loss

    def forward_decoder(self,x,series_embed):

        #普通decoder，使用单输入流，单层、未改造的单层transformer block作为decoder.
        x_label=x
        #normalization
        if self.is_norm:
            x,means,stdev=instance_norm(x)

        pos_embed=self.PositionEmbedding(torch.zeros(x.shape[0],self.input_len+self.series_embed_len))
        if(self.train_pe):
            x=self.ScalarProjection_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
        else:
            x=self.ScalarProjection_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)

        x_masked, mask, ids_restore=self.mask_func(x,self.mask_rate_dec)
        #添加series_embedding
        x=torch.cat([series_embed,x_masked],dim=1)

        #输入decoder
        x=self.Transformer_Decoder(x.transpose(0,1)).transpose(0,1)

        series_embed=x[:,:self.series_embed_len,:]
        #重新排列
        mask_tokens=nn.Parameter(torch.zeros(ids_restore.shape[0],int(self.input_len*self.mask_rate_dec)+1,x.shape[2])).to(x.device)
        x=torch.cat([x[:,self.series_embed_len:,:],mask_tokens],dim=1)

        x=torch.gather(x,dim=1,index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))#复原被打乱的x_mask序列
        
        #embedding & add cls_token
        #x=self.ScalarProjection_pred_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
        x=torch.cat([series_embed,x],dim=1)


        #预测
        x=self.decoder_proj(x)
        x=self.decoder_pred(x.permute(0,2,1).to(torch.float32)).transpose(1,2)

        #denormalization
        if self.is_norm:
            x=instance_denorm(x,means,stdev,self.input_len)

        loss_dec=self.forward_loss(x_label,x,mask)

        return loss_dec,x
    
    def forward_enhanced_decoder(self,x,series_embed):
        x_label=x
        #normalization
        if self.is_norm:
            x,means,stdev=instance_norm(x)

        pos_embed=self.PositionEmbedding(torch.zeros(x.shape[0],self.input_len+self.series_embed_len))
        series_embed_exp=series_embed.repeat(1,int((self.input_len+self.series_embed_len)/self.series_embed_len),1)

        if(self.train_pe):
            x=self.ScalarProjection_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
            h1=pos_embed+series_embed_exp
        else:
            x=self.ScalarProjection_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
            h1=series_embed_exp

        #x=self.ScalarProjection_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]

        h2=torch.cat([series_embed,x],dim=1)
        #h1=h2
        #print(h1.shape)
        
        attn_mask=self.attn_mask(h1,self.mask_rate_dec)
        #print(attn_mask)

        x=self.EnhancedDecoderLayer(h1,h2,attn_mask)

        #预测
        #x=self.ScalarProjection_pred_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed

        x=self.decoder_proj(x)
        x=self.decoder_pred(x.permute(0,2,1).to(torch.float32)).transpose(1,2)

        #denormalization
        if self.is_norm:
            x=instance_denorm(x,means,stdev,self.input_len)
        #loss
        loss_dec=self.criterion(x,x_label)

        return loss_dec,x
        
    def forward(self,x,series_embed):
        if self.enhance_decoding:
            loss_dec,x=self.forward_enhanced_decoder(x,series_embed)
        else:
            loss_dec,x=self.forward_decoder(x,series_embed)

        return loss_dec,x


class RetroMaskedAutoEncoder(nn.Module):
    def __init__(self,c_in,d_model,input_len,series_embed_len,mask_rate_enc,mask_rate_dec,encoder_depth,mask_size,enhance_decoding=False,
                 alpha=1.0,train_pe=True,is_norm=False
        ):
        super(RetroMaskedAutoEncoder, self).__init__()
        self.alpha=alpha
        #encoder
        self.encoder=RetroMAE_Encoder(c_in=c_in,d_model=d_model,input_len=input_len,series_embed_len=series_embed_len,mask_rate_enc=mask_rate_enc,
                encoder_depth=encoder_depth,mask_size=mask_size,train_pe=train_pe,is_norm=is_norm)
        #decoder
        self.decoder=RetroMAE_Decoder(c_in=c_in,d_model=d_model,input_len=input_len,series_embed_len=series_embed_len,mask_rate_dec=mask_rate_dec,
                mask_size=mask_size,enhance_decoding=enhance_decoding,train_pe=train_pe,is_norm=is_norm)

    def repr_gen(self,x,pe,fine_tune=False):#生成表征
        x=self.encoder.repr_gen(x,pe,fine_tune=fine_tune)
        return x
    
    def dim_loss(self,pred_label,pred,dim):#返回每个维度的损失
        loss=self.encoder.dim_loss(pred_label,pred,dim)+self.decoder.dim_loss(pred_label,pred,dim)
        return loss
    
    def forward(self,x):
        #instance_norm
        '''means = torch.sum(x, dim=1) / torch.sum( dim=1)
        means = means.unsqueeze(1).detach()
        x = x - means
        x = x.masked_fill(0)
        stdev = torch.sqrt(torch.sum(x * x, dim=1) /
                           torch.sum(dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x /= stdev'''

        series_embed,loss_enc,enc_out=self.encoder(x)
        loss_dec,dec_out=self.decoder(x,series_embed)
        '''print("enc_loss:{}".format(loss_enc))
        print("dec_loss:{}".format(loss_dec))'''
        loss=(loss_enc+loss_dec*self.alpha)/(1+self.alpha)
        loss=loss.to(torch.float32)
        #loss=loss_dec
        #print(loss)
        return enc_out,loss



