import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from Embedding import PatchEmbedding,PositionalEmbedding,Embedding,Patching,TokenEmbedding
from RetroMAElike_model import instance_denorm,instance_norm
from Transformer_Layers import MultiDecompTransformerEncoderLayer,ResidualDecompEncoder,Encoder,EncoderLayer
from attention import AttentionLayer,FullAttention


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, loss1,loss2):
        self.params = self.params.to(loss1.device)
        loss_sum=0
        loss_sum += 0.5 / (self.params[0] ** 2) * loss1 + torch.log(1 + self.params[0] ** 2)
        loss_sum += 0.5 / (self.params[1] ** 2) * loss2 + torch.log(1 + self.params[1] ** 2)
            #loss_sum += 0.5 / (self.params[i] ** 2) 
        return loss_sum

class SimMTM(nn.Module):
    def __init__(self,mask_rate,window_size,encoder_depth,c_in,d_model,n_head,input_len,is_norm,time_block,window_list=[1],CI=False,mask_num=2,tau=0.2,is_decomp=True,
                 backbone='res',part='s'):
        super(SimMTM, self).__init__()
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
        self.mask_num=mask_num
        self.tau=tau
        self.is_decomp=is_decomp
        self.CI=CI
        self.part=part
        
        self.series_decomp=series_decomp(kernel_size=window_size)
        self.embedding_enc_s=TokenEmbedding(c_in=1 if CI else c_in,d_model=d_model)
        self.embedding_enc_t=Embedding(c_in=1 if CI else c_in,d_model=d_model,c_num=c_in,CI=CI,pos_embed=False)

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
        Transformer_Encoder_Layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head,norm_first=True,batch_first=True)
        self.encoder_t=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head,norm_first=True,batch_first=True)
        #self.encoder=MultiDecompTransformerEncoderLayer(c_in=c_in,d_model=d_model,input_len=input_len)
        self.Patching=Patching(patch_len=time_block)


        self.dec_conv_s=nn.Conv1d(d_model,d_model,kernel_size=3,stride=1,padding=1,padding_mode='circular')
        self.dec_conv_t=nn.Conv1d(d_model,d_model,kernel_size=3,stride=1,padding=1,padding_mode='circular')
        #self.decoder_s=ChannelIndependenceTransformerEncoderLayer(c_in=1,d_model=d_model) if CI else nn.TransformerEncoderLayer(d_model=d_model,nhead=4,norm_first=True,batch_first=True)
        #self.decoder_t=ChannelIndependenceTransformerEncoderLayer(c_in=1,d_model=d_model) if CI else nn.TransformerEncoderLayer(d_model=d_model,nhead=4,norm_first=True,batch_first=True)

        self.proj_s=nn.Linear(in_features=d_model,out_features=1 if CI else c_in)
        self.proj_t=nn.Linear(in_features=d_model,out_features=1 if CI else c_in)
        self.is_norm=is_norm
        self.mse_loss=nn.MSELoss()
        self.auto_loss=AutomaticWeightedLoss()
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
    
    def generate_mask_series(self,x,mask_num,is_rand=True): # x: (B*nvar)*L*1 每个单通道序列生成mask_num个掩码序列 返回mask位置填充0的序列

        for i in range(0,mask_num):
            if i==0:
                #x_masked,mask,ids_restore=self.mask_func_block(x,mask_ratio=self.mask_rate,is_rand=is_rand)
                x_masked,mask,ids_restore=self.mask_func(x,mask_ratio=self.mask_rate)
                #print(x_masked.shape)
            else:
                #x_masked_,mask_,ids_restore_=self.mask_func_block(x,mask_ratio=self.mask_rate,is_rand=is_rand)
                x_masked_,mask_,ids_restore_=self.mask_func(x,mask_ratio=self.mask_rate)
                x_masked=torch.concat([x_masked,x_masked_],dim=0)
                mask=torch.concat([mask,mask_],dim=0)
                ids_restore=torch.concat([ids_restore,ids_restore_],dim=0)
                
        mask_tokens=nn.Parameter(torch.zeros(ids_restore.shape[0],int(self.input_len*self.mask_rate),1 if self.CI else self.c_in)).to(x.device)

        x_masked=torch.cat([x_masked,mask_tokens],dim=1)
        x_masked=torch.gather(x_masked,dim=1,index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))#复原被打乱的x_mask序列
            
        return x_masked
    
    def instance_contrastive_loss(self,z1, z2):
        B, T = z1.size(0), z1.size(1)
        if B == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        sim = sim/self.tau
        #sim=F.cosine_similarity(z, z.transpose(1, 2))
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)  -1代表最后一个？ 
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]     #diagonal=-1:对角线为0，=1::对角线为原值。对角线(aii,i<=min(d1,d2))
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def series_contrastive_learning(self,x,mask_num):
        num,_,_=x.shape
        num=num//(mask_num+1)
        
        contrastive_loss=0
        x0=x[:num,:,:]

        for i in range(0,mask_num):
            x_=x[(num*(i+1)):(num*(i+2)),:,:]
            loss=self.instance_contrastive_loss(x0,x_)
            #print(loss)
            contrastive_loss=contrastive_loss+loss
        
        contrastive_loss=contrastive_loss/mask_num
        return contrastive_loss

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
        #embed
        x_s_,x_t_=self.series_decomp(x)#x_s_,x_t_分别作为两部分的ground truth
        x_t=x_t_

        if self.is_norm:
            if not self.is_decomp:
                x_norm,means,stdev=instance_norm(x)
                x_s=x_norm
            else:
                x_t,means,stdev=instance_norm(x_t)
                x_s=x_s_ if self.part=='s' else x_t
            #means=self.mean_linear(means)
            #stdev=self.stdev_linear(stdev)
        else:
            x_s=x

        #x_s,x_t=self.series_decomp(x_norm)
        #reshape -> CI

        if self.CI:
            x_s=x_s.permute(0,2,1)
            x_s=torch.reshape(x_s,(x_s.shape[0]*x_s.shape[1],x_s.shape[2])).unsqueeze(-1)
        
        '''if self.is_norm:
            x_t,means,stdev=instance_norm(x_t_)'''
       
        #print(x_s.shape)
        x_s_masked=self.generate_mask_series(x_s,self.mask_num,True)

        x_s=torch.concat([x_s,x_s_masked],dim=0)

        x_s=self.embedding_enc_s(x_s)

        x_s=self.encoder_s(x_s)

        #patching ,contrastive learining
        x_s_patch=self.Patching(x_s)#x:(B*nvar)*patch_num*d_model
        x_s_patch=F.normalize(x_s_patch,dim=-1)
        contrastive_loss=self.series_contrastive_learning(x_s_patch,self.mask_num)
        #x_s_patch:(B*nvar)*patch_num*d_model as patch_embed

        #aggregation
        patch_num=x_s_patch.shape[0]*x_s_patch.shape[1]//(self.mask_num+1)
        #x_s_patch=F.normalize(x_s_patch,dim=-1)
        x_s_patch=torch.reshape(x_s_patch,(x_s_patch.shape[0]*x_s_patch.shape[1],1,x_s_patch.shape[2])).squeeze(1)
        #print(x_s_patch.shape,patch_num)
        x_s=torch.reshape(x_s,(x_s.shape[0]*(self.input_len//self.time_block),self.time_block,x_s.shape[2]))
        #patch similarity
        x_s_patch0=x_s_patch[:patch_num,:]

        for i in range(0,self.mask_num+1):
            x_s_patch_=x_s_patch[patch_num*i:patch_num*(i+1),:]
            #sim_matrix_=F.cosine_similarity(x_s_patch0.unsqueeze(0),x_s_patch_.unsqueeze(1),dim=2)
            sim_matrix_=torch.matmul(x_s_patch0,x_s_patch_.permute(1,0))
            if i==0:
                sim_matrix=sim_matrix_
            else:
                sim_matrix=torch.concat([sim_matrix,sim_matrix_],dim=1)

        #sim_matrix=F.cosine_similarity(x_s_patch.unsqueeze(0),x_s_patch.unsqueeze(1),dim=2)
        sim_matrix=torch.exp(sim_matrix/self.tau)
        x_s_recons=nn.Parameter(torch.zeros(x_s.shape[0]//(self.mask_num+1),x_s.shape[1]*x_s.shape[2])).to(x_s.device)
        #print(x_s_recons.shape)
        ones=torch.ones(sim_matrix.shape).to(x.device)
        for i in range(0,sim_matrix.shape[0]):
            ones[i,i]=0
        sim_matrix=sim_matrix*ones
        #print(sim_matrix.shape)

        numerator=sim_matrix.permute(1,0)
        denominator=torch.sum(sim_matrix,dim=1)
        score=(numerator/denominator).permute(1,0)
        score=score[:x_s_recons.shape[0],:]
        #print(score.shape,torch.reshape(x_s,(x_s.shape[0],x_s.shape[1]*x_s.shape[2])).shape)

        '''for i in range(0,score.shape[0]):
            #print(score[i,:].shape)
            x_score=score[i,:].reshape(score.shape[1],1)*torch.reshape(x_s,(x_s.shape[0],x_s.shape[1]*x_s.shape[2]))
            x_s_recons[i,:]=torch.sum(x_score,dim=0)'''
        x_s_recons=torch.matmul(score,torch.reshape(x_s,(x_s.shape[0],x_s.shape[1]*x_s.shape[2])))
        x_s_recons=torch.reshape(x_s_recons,(x_s_recons.shape[0],x_s_recons.shape[1]//self.d_model,self.d_model))
        x_s_recons=torch.reshape(x_s_recons,(x_s_recons.shape[0]//(self.input_len//self.time_block),x_s_recons.shape[1]*(self.input_len//self.time_block),x_s_recons.shape[2]))
        
        #dec_embed
        x_s=self.dec_conv_s(x_s_recons.permute(0,2,1).float()).transpose(1,2)
        #x_s=x_s        
        x_s=self.proj_s(x_s)
        if self.CI:
            x_s=torch.reshape(x_s,(x_s.shape[0]//self.c_in,self.c_in,x_s.shape[1]))
            x_s=x_s.permute(0,2,1)

        #print(x_s.shape,x_s_.shape)
        if self.is_norm and((not self.is_decomp) or self.part=='t'):
            x_s=instance_denorm(x_s,means,stdev,seq_len=self.input_len)

        #计算loss
        if self.is_decomp:
            loss_s=self.mse_loss(x_s,x_s_) if self.part=='s' else self.mse_loss(x_s,x_t_)
        else:
            loss_s=self.mse_loss(x_s,x)
        #loss_tuple=loss_s,contrastive_loss
        loss=self.auto_loss(loss_s,contrastive_loss)

        print('loss_s:{}'.format(loss_s))
        print('loss_c:{}'.format(contrastive_loss))

        return x_s,loss

    def repr_gen(self,x_s,x_t,frozen_num,is_ft=True):


        x_s=self.embedding_enc_s(x_s)
        x_t=self.embedding_enc_t(x_t)

        x_s=self.encoder_s(x_s)
            
        x_t=self.encoder_t(x_t)

        return x_s,x_t


    





