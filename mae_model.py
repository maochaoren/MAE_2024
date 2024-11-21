import torch.nn as nn
import torch
import torch.nn.functional as F
from Embedding import PatchEmbedding,PositionalEmbedding


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


class MaskedAutoEncoder(nn.Module):
    def __init__(self,c_in,d_model,input_len,mask_rate,encoder_depth,decoder_depth,mask_size,freq,
        train_pe=True,encoder='trans'):
        super(MaskedAutoEncoder, self).__init__()
        self.d_model=d_model
        self.input_len=input_len
        self.mask_rate=mask_rate
        #self.embed_dim=embed_dim
        self.c_in=c_in
        self.mask_size=mask_size
        self.mask_token=nn.Parameter(torch.zeros(1,1,d_model))

        #encoder
        self.encoder=encoder
        #embed
        self.train_pe=train_pe
        self.ScalarProjection_enc=nn.Conv1d(c_in,d_model,kernel_size=3,padding=1,padding_mode='circular')
        self.PositionEmbedding=PositionalEmbedding(d_model=d_model)
        #transformer block encoder
        Transformer_Encoder_Layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=1,norm_first=True,batch_first=False)
        self.Transformer_Encoder=nn.TransformerEncoder(Transformer_Encoder_Layer,num_layers=encoder_depth)
        #dilated conv encoder
        self.DilatedConv_Encoder = DilatedConvEncoder(
            d_model,
            [d_model] * encoder_depth + [d_model],
            kernel_size=3
        )
        self.encoder_norm_layer=nn.LayerNorm(d_model)
        #decoder
        self.ScalarProjection_dec=nn.Conv1d(d_model,d_model,kernel_size=3,padding=1,padding_mode='circular')
        #transformer decoder
        Transformer_Decoder_Layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=1,norm_first=False,batch_first=False)
        self.Transformer_Decoder=nn.TransformerEncoder(Transformer_Decoder_Layer,num_layers=decoder_depth)
        self.decoder_pred=nn.Linear(d_model,c_in,bias=True)
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


    '''def mask_func(self,X_pte,mask_ratio):
        x=X_pte

        N, _, D = x.shape # batch, length(num_patches), dim
        L=int(self.input_len/self.mask_size)
        len_keep = int(L * (1 - mask_ratio))

        mask_rand=torch.rand(N,L,device=x.device)
        ids_shuffle = torch.argsort(mask_rand, dim=1)  # ascend: small is keep, large is remove(和mask_ratio相比) 其实是第0维（同一行）之间排序
        ids_restore = torch.argsort(ids_shuffle, dim=1)#两次ascend排序并取下标，得到原本Noise的相对大小（原本的值越大，ids_restore中对应位置的值就越大）
        #print(ids_shuffle[0])
        #小的留下，大的mask掉
        ids_keep = ids_shuffle[:, :len_keep]
        #print(ids_restore.shape)

        #增广ids_keep
        ids_keep_series=torch.zeros((N,len_keep*self.mask_size),device=x.device)
        for i in range(0,len_keep):
            for j in range(0,self.mask_size):
                ids_keep_series[:,i*self.mask_size+j]=ids_keep[:,i]*self.mask_size+j

        x_masked = torch.gather(x, dim=1, index=ids_keep_series.unsqueeze(-1).repeat(1, 1, D).long())#gather:按照给定的idx挑选第dim维保留的元素，然后将第dim维消去。
        #print(x_masked.shape)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0 #在mask数组中，0表示未被Mask，1表示被Mask的
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        #增广Mask，长度变为mask_size倍
        mask_series=torch.zeros((N,self.mask_size*L),device=x.device)
        for i in range(0,L):
            mask_series[:,i*self.mask_size:(i+1)*self.mask_size]=mask[:,i].unsqueeze(1)

        #增广ids_restore方法,同ids_keep
        ids_restore_series=torch.zeros((N,self.mask_size*L),device=x.device)
        for i in range(0,L):
            for j in range(0,self.mask_size):
                ids_restore_series[:,i*self.mask_size+j]=ids_restore[:,i]*self.mask_size+j
        return x_masked, mask_series, ids_restore_series.long()'''
    
    def repr_gen(self,x,pe,fine_tune):#生成表征
        pos_embed=self.PositionEmbedding(torch.zeros(x.shape[0],self.input_len))
        if fine_tune:
            if(pe):
                x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed
            else:
                x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)

            if self.encoder=='trans':
                x=self.Transformer_Encoder(x.transpose(0,1)).transpose(0,1)
            elif self.encoder=='conv':
                x=self.DilatedConv_Encoder(x.transpose(1,2)).transpose(1,2)
            return x
        else:
            with torch.no_grad():
                if(pe):
                    x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed
                else:
                    x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
    
                if self.encoder=='trans':
                    x=self.Transformer_Encoder(x.transpose(0,1)).transpose(0,1)
                elif self.encoder=='conv':
                    x=self.DilatedConv_Encoder(x.transpose(1,2)).transpose(1,2)
                return x
    
    def mask_func(self,x,mask_ratio):
        
        N, _, D = x.shape # batch, length(num_patches), dim
        L=int(self.input_len)
        len_keep = int(L * (1 - mask_ratio))

        mask_rand=torch.rand(N,L,device=x.device)
        ids_shuffle = torch.argsort(mask_rand, dim=1)  # ascend: small is keep, large is remove(和mask_ratio相比) 其实是第0维（同一行）之间排序
        ids_keep_restore = torch.sort(ids_shuffle[:,:len_keep], dim=1).values
        ids_restore=torch.argsort(torch.cat([ids_keep_restore,ids_shuffle[:,len_keep:]],dim=1),dim=1)#不改变保留token的相对次序
        #小的留下，大的mask掉
        x_masked = torch.gather(x, dim=1, index=ids_keep_restore.unsqueeze(-1).repeat(1, 1, D).long())#gather:按照给定的idx挑选第dim维保留的元素，然后将第dim维消去。
        #print(x_masked.shape)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0 #在mask数组中，0表示未被Mask，1表示被Mask的
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore.long()


    #def patchify(self,x):

    def forward_encoder(self,x): 
        pos_embed=self.PositionEmbedding(torch.zeros(x.shape[0],self.input_len))
        #print(x.shape)
        if(self.train_pe):
            x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed
        else:
            x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
        x_masked, mask, ids_restore=self.mask_func(x,self.mask_rate)#mask之前已经被patch化
        #encoder_in=self.mask_cat(x_masked,mask)
        #x_masked=self.encoder_norm_layer(x_masked)
        if self.encoder=='trans':
            x=self.Transformer_Encoder(x_masked.transpose(0,1)).transpose(0,1)
        elif self.encoder=='conv':
            x=self.DilatedConv_Encoder(x_masked.transpose(1,2)).transpose(1,2)

        self.mask=mask
        
        return x,mask,ids_restore

    def forward_Rdecoder(self,x,ids_restore):
        pos_embed=self.PositionEmbedding(torch.zeros(x.shape[0],self.input_len))

        mask_tokens=self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        #print(x.shape)
        x_=torch.cat([x,mask_tokens],dim=1)
        #print(x_.shape)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        #print(x_.shape)
        #x_=self.PatchEmbedding_dec(x_)
        if(self.train_pe):
            x=self.ScalarProjection_dec(x_.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed
        else:
            x=self.ScalarProjection_dec(x_.permute(0,2,1).to(torch.float32)).transpose(1,2)
        #x=self.decoder_norm_layer(x)
        x=self.Transformer_Decoder(x.transpose(0,1)).transpose(0,1)
        x=self.decoder_pred(x)
        #x=self.dropout(x)
        return x
    
    def forward_loss(self,pred_label,pred,mask):
        x=pred
        #print(x[:,:,0])
        #计算MSE Loss
        loss = (x-pred_label) ** 2
        #print(loss[:,:,0])
        #print(x)
        #print(loss[:,:,0])
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
        '''if(dim==0):
            print(loss)'''
        #loss = loss.mean(dim=-1) #loss:B x (L x P)
        #print(loss*self.mask)
        loss=(loss*self.mask).sum()/self.mask.sum()
        return loss
    
    def forward(self,x):
        latent,mask,ids_restore=self.forward_encoder(x)
        pred=self.forward_Rdecoder(latent,ids_restore)
        #pred=pred.reshape(shape=(pred.shape[0],pred.shape[1]*self.patch_size,self.c_in))
        loss=self.forward_loss(pred_label=x,pred=pred,mask=mask)
        return pred,loss
        #return pred,mask



