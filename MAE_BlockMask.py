import torch.nn as nn
import torch
from Embedding import PatchEmbedding,PositionalEmbedding

#时间序列先切成定长（如一天时间），再在每一段上按照比例进行mask
class BlockMaskedAutoEncoder(nn.Module):
    def __init__(self,c_in,d_model,input_len,mask_rate,encoder_depth,decoder_depth,time_block,freq
        ):
        super(BlockMaskedAutoEncoder, self).__init__()
        self.d_model=d_model
        self.input_len=input_len
        self.mask_rate=mask_rate
        #self.embed_dim=embed_dim
        self.c_in=c_in
        self.time_block=time_block
        self.mask_token=nn.Parameter(torch.zeros(1,1,d_model))
        #encoder
        self.PatchEmbedding_enc=PatchEmbedding(d_model=d_model,c_in=c_in,input_len=input_len,patch_size=time_block,freq=freq,patching=False)
        #self.num_patches=self.PatchEmbedding.num_patches#一个序列有几个patch
        Transformer_Encoder_Layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=4)
        self.Transformer_Encoder=nn.TransformerEncoder(Transformer_Encoder_Layer,num_layers=encoder_depth)
        self.encoder_norm_layer=nn.LayerNorm(d_model)
        #decoder
        self.PatchEmbedding_dec=PatchEmbedding(d_model=d_model,c_in=d_model,input_len=input_len,patch_size=time_block,freq=freq,patching=False)
        Transformer_Decoder_Layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=4)
        self.Transformer_Decoder=nn.TransformerEncoder(Transformer_Decoder_Layer,num_layers=decoder_depth)
        self.decoder_norm_layer=nn.LayerNorm(d_model)
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


    def mask_func(self,X_pte,mask_ratio):
        x=X_pte

        N, _, D = x.shape # batch, length(num_patches), dim
        #L=int(self.input_len/self.time_block)
        len_keep = int(self.time_block * (1 - mask_ratio))

        ids_restore=torch.zeros((N,self.input_len),device=x.device)
        ids_keep=torch.zeros((N,int(self.input_len*(1-mask_ratio))),device=x.device)
        mask = torch.ones([N, self.input_len], device=x.device)
        x_masked=torch.zeros((N,int(self.input_len*(1-mask_ratio)),D),device=x.device)

        for i in range(0,int(self.input_len/self.time_block)):          
            mask_rand=torch.rand(N,self.time_block,device=x.device)
            ids_shuffle = torch.argsort(mask_rand, dim=1)  # ascend: small is keep, large is remove(和mask_ratio相比) 其实是第0维（同一行）之间排序
            ids_restore_block = torch.argsort(ids_shuffle, dim=1)#两次ascend排序并取下标，得到原本Noise的相对大小（原本的值越大，ids_restore中对应位置的值就越大）
            mask_block=torch.ones([N, self.time_block], device=x.device)
            mask_block[:,:len_keep]=0
            mask_block=torch.gather(mask_block, dim=1, index=ids_restore_block.long())

            #ids_restore_block+=i*self.time_block

            #小的留下，大的mask掉
            ids_keep_block = ids_shuffle[:, :len_keep]
            x_masked_block=torch.gather(x, dim=1, index=ids_keep_block.unsqueeze(-1).repeat(1, 1, D).long())#gather:按照给定的idx挑选第dim维保留的元素，然后将第dim维消去。
            ids_keep_block+=i*self.time_block

            ids_keep[:,int(i*self.time_block*(1-mask_ratio)):int((i+1)*self.time_block*(1-mask_ratio))]=ids_keep_block
            ids_restore[:,i*self.time_block:(i+1)*self.time_block]=ids_restore_block
            mask[:,i*self.time_block:(i+1)*self.time_block]=mask_block
            x_masked[:,int(i*self.time_block*(1-mask_ratio)):int((i+1)*self.time_block*(1-mask_ratio)),:]=x_masked_block
        #print(ids_shuffle[0])
        #print(ids_restore.shape)

        #print(mask[0])
        #x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D).long())#gather:按照给定的idx挑选第dim维保留的元素，然后将第dim维消去。
        #print(x_masked.shape)
        '''mask[:, :int(self.input_len * (1 - mask_ratio))] = 0 #在mask数组中，0表示未被Mask，1表示被Mask的
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore.long())'''
        return x_masked, mask, ids_restore.long()
    


    #def patchify(self,x):

    def forward_encoder(self,x):  
        X_pte=self.PatchEmbedding_enc(x)
        x_masked, mask, ids_restore=self.mask_func(X_pte,self.mask_rate)#mask之前已经被patch化
        #encoder_in=self.mask_cat(x_masked,mask)
        x_masked=self.encoder_norm_layer(x_masked)
        x=self.Transformer_Encoder(x_masked)

        

        return x,mask,ids_restore

    def forward_Rdecoder(self,x,ids_restore):
        
        #mask_tokens=self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        #print(mask_tokens.shape)
        #print(x.shape)
        x_=nn.Parameter(torch.zeros((x.shape[0],1,x.shape[2]),device=x.device))

        for i in range(0,int(self.input_len/self.time_block)):
            x_block=x[:,int(i*self.time_block*(1-self.mask_rate)):int((i+1)*self.time_block*(1-self.mask_rate)),:]
            mask_tokens_block=nn.Parameter(torch.zeros((x.shape[0],int(self.time_block*self.mask_rate),x.shape[2]),device=x.device))
            x_block_concat=torch.cat([x_block,mask_tokens_block],dim=1)
            #print(ids_restore[:,i*self.time_block:(i+1)*self.time_block].unsqueeze(-1).repeat(1, 1, x.shape[2]).shape)
            #print(x_block_concat.shape)
            x_block_concat=torch.gather(x_block_concat, dim=1, index=ids_restore[:,i*self.time_block:(i+1)*self.time_block].unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x_=torch.concat([x_,x_block_concat],dim=1)
        x_=x_[:,1:,:]
        '''x_=torch.cat([x,mask_tokens],dim=1)
        #print(x_.shape)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))'''  
        #print(x_.shape)
        x_=self.PatchEmbedding_dec(x_)
        x_=self.decoder_norm_layer(x_)
        x=self.Transformer_Decoder(x_)
        #x=self.decoder_norm_layer(x)

        x=self.decoder_pred(x)
        #x=self.dropout(x)
        return x
    
    def forward_loss(self,x_label,pred,mask):
        #pred B x L x (patch_size x c_in) 先 -> B x (L x P) x c_in
        #print(pred.shape)
        L=pred.shape[1]
        C=self.c_in
        #print(pred.shape)
        x=pred
        #print(x.shape)
        #计算MSE Loss
        #print(mask)
        loss = (x-x_label) ** 2
        #print(x)
        loss = loss.mean(dim=-1) #loss:B x (L x P)
        loss=(loss*mask).sum()/mask.sum()
        return loss
    
    def forward(self,x,x_mask):
        latent,mask,ids_restore=self.forward_encoder(x)
        pred=self.forward_Rdecoder(latent,ids_restore)
        #pred=pred.reshape(shape=(pred.shape[0],pred.shape[1]*self.patch_size,self.c_in))
        loss=self.forward_loss(x_label=x,pred=pred,mask=mask)
        return loss,pred,mask
        #return pred,mask



