import torch.nn as nn
import torch.nn.functional as F
import torch
from Embedding import PatchEmbedding,PositionalEmbedding
from attention import DualInputTransformerEncoder 


class RetroMaskedAutoEncoder(nn.Module):
    def __init__(self,c_in,d_model,input_len,series_embed_len,mask_rate_enc,mask_rate_dec,encoder_depth,decoder_depth,mask_size,enhance_decoding=False,
                 alpha=1.0,train_pe=True
        ):
        super(RetroMaskedAutoEncoder, self).__init__()
        self.alpha=alpha
        self.d_model=d_model
        self.input_len=input_len
        self.series_embed_len=series_embed_len
        self.mask_rate_enc=mask_rate_enc
        self.mask_rate_dec=mask_rate_dec
        self.enhance_decoding=enhance_decoding
        #self.embed_dim=embed_dim
        self.c_in=c_in
        self.mask_size=mask_size
        self.mask_token=nn.Parameter(torch.zeros(1,1,d_model))
        #encoder
        self.train_pe=train_pe
        self.ScalarProjection_enc=nn.Conv1d(c_in,d_model,kernel_size=3,padding=1,padding_mode='circular')
        self.PositionEmbedding=PositionalEmbedding(d_model=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1,self.series_embed_len,d_model))
        #self.num_patches=self.PatchEmbedding.num_patches#一个序列有几个patch
        Transformer_Encoder_Layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=4,norm_first=True,batch_first=False)
        self.Transformer_Encoder=nn.TransformerEncoder(Transformer_Encoder_Layer,num_layers=encoder_depth)


        self.encoder_proj=nn.Linear(d_model,c_in,bias=True)
        self.encoder_pred=nn.Linear(input_len+series_embed_len,input_len,bias=True)
        #decoder
        self.ScalarProjection_dec=nn.Conv1d(c_in,d_model,kernel_size=3,padding=1,padding_mode='circular')
        Transformer_Decoder_Layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=4,norm_first=True,batch_first=False)
        self.Transformer_Decoder=nn.TransformerEncoder(Transformer_Decoder_Layer,num_layers=decoder_depth)
        self.decoder_norm_layer=nn.LayerNorm(d_model)
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
    
        #mask_rand=
        
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
    

    def repr_gen(self,x,pe):#生成表征
        pos_embed=self.PositionEmbedding(torch.zeros(x.shape[0],self.input_len+self.series_embed_len))
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
    
    def forward_encoder(self,x): 
        x_label=x
        
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

        loss_enc=self.forward_loss(x_label,x,mask)
        return series_embed,loss_enc,x

    def forward_decoder(self,x,series_embed):

        #普通decoder，使用单输入流，单层、未改造的单层transformer block作为decoder.
        x_label=x
        pos_embed=self.PositionEmbedding(torch.zeros(x.shape[0],self.input_len+self.series_embed_len))
        if(self.train_pe):
            x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
        else:
            x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)

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
        x=self.ScalarProjection_pred_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
        x=torch.cat([series_embed,x],dim=1)

        #预测
        x=self.decoder_proj(x)
        x=self.decoder_pred(x.permute(0,2,1).to(torch.float32)).transpose(1,2)

        loss_dec=self.forward_loss(x_label,x,mask)

        return loss_dec,x
    
    def forward_enhanced_decoder(self,x,series_embed):
        x_label=x
        pos_embed=self.PositionEmbedding(torch.zeros(x.shape[0],self.input_len+self.series_embed_len))
        series_embed_exp=series_embed.repeat(1,int((self.input_len+self.series_embed_len)/self.series_embed_len),1)

        if(self.train_pe):
            x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
            h1=pos_embed+series_embed_exp
        else:
            x=self.ScalarProjection_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
            h1=series_embed_exp

        #x=self.ScalarProjection_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]

        h2=torch.cat([series_embed,x],dim=1)
        #h1=h2
        #print(h1.shape)
        
        attn_mask=self.attn_mask(h1,self.mask_rate_dec)
        #print(attn_mask)

        x=self.EnhancedDecoderLayer(h1,h2,attn_mask)

        #预测
        x=self.ScalarProjection_pred_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed

        x=self.decoder_proj(x)
        x=self.decoder_pred(x.permute(0,2,1).to(torch.float32)).transpose(1,2)

        #loss
        loss=self.criterion(x,x_label)

        return loss,x



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
    
    def forward(self,x):
        series_embed,loss_enc,enc_out=self.forward_encoder(x)
        if self.enhance_decoding:
            loss_dec,dec_out=self.forward_enhanced_decoder(x,series_embed)
        else:
            loss_dec,dec_out=self.forward_decoder(x,series_embed)
        loss_dec*=self.alpha
        '''print("enc_loss:{}".format(loss_enc))
        print("dec_loss:{}".format(loss_dec))'''
        loss=(loss_enc+loss_dec)/(1+self.alpha)
        loss=loss.to(torch.float32)
        #loss=loss_dec
        #print(loss)
        return enc_out,loss
