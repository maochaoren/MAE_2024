import torch.nn as nn
import torch
from repres_eval import _eval_with_pooling
from Embedding import PatchEmbedding,PositionalEmbedding

class FineTuningModelCLS(nn.Module):
    def __init__(self,c_in,d_model,input_len,series_embed_len,mae_encoder,mae_project_enc,encoder_depth,mask_size,cls_num,freq='h',is_mae=True,use_else_tokens=False,
                 train_pe=True,use_cls_token=True
        ):
        super(FineTuningModelCLS, self).__init__()
        self.use_cls_token=use_cls_token
        self.d_model=d_model
        self.series_embed_len=series_embed_len
        self.input_len=input_len
        self.use_else_tokens=use_else_tokens
        self.c_in=c_in
        self.mask_size=mask_size
        self.cls_num=cls_num
        #self.input_len=input_len
        self.cls_token = nn.Parameter(torch.zeros(1, series_embed_len, d_model))
        self.PositionalEmbedding=PositionalEmbedding(d_model=d_model)
        
        #encoder
        self.train_pe=train_pe
        self.proj_enc=mae_project_enc if is_mae else nn.Conv1d(c_in,d_model,kernel_size=3,padding=1,padding_mode='circular')
        self.pos_embed_enc=PositionalEmbedding(d_model=d_model)
        #self.num_patches=self.PatchEmbedding.num_patches
        #self.PatchEmbedding_x_label=PatchEmbedding(d_model=d_model,input_len=label_len,c_in=c_in,patch_size=mask_size)
        #self.NonPatchEmbedding=PatchEmbedding(d_model=d_model,input_len=self.label_len+self.pred_len,c_in=c_in,patch_size=mask_size,patching=False)
        Transformer_Encoder_Layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=4,norm_first=True)
        self.encoder=mae_encoder if is_mae else nn.TransformerEncoder(Transformer_Encoder_Layer,num_layers=encoder_depth)
        self.encoder_norm_layer=nn.LayerNorm(d_model)
        self.is_mae=is_mae
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        #decoder
        self.proj_dec=nn.Conv1d(d_model,d_model,kernel_size=3,padding=1,padding_mode='circular')
        self.decoder_proj=nn.Linear(d_model,c_in,bias=True)
        if self.use_cls_token:
            self.decoder_pred=nn.Linear((self.input_len+series_embed_len)*d_model,cls_num) if use_else_tokens else nn.Linear(series_embed_len*d_model,cls_num)
        else:
            self.decoder_pred=nn.Linear(1*d_model,cls_num)
        self.sigmoid=nn.Sigmoid()
        #self.dropout=nn.Dropout(p=0.1)


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

    def repr_gen(self,x,pe):#生成表征
        pos_embed=self.PositionalEmbedding(torch.zeros(x.shape[0],self.input_len+self.series_embed_len))
        with torch.no_grad():
            if(pe):
                x=self.proj_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
            else:
                x=self.proj_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
            
            #添加cls_token，cls_token生成的embedding作为整个时间序列的embedding
            cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
            #cls_tokens = cls_tokens+pos_embed[:,:self.series_embed_len]
            
            x=torch.cat([cls_tokens,x],dim=1)
    
            x=self.encoder(x.transpose(0,1)).transpose(0,1)
            return x


    def forward_encoder(self,x):  
        pos_embed=self.PositionalEmbedding(torch.zeros(x.shape[0],self.input_len+self.series_embed_len))
        if(self.train_pe):
            x=self.proj_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)+pos_embed[:,self.series_embed_len:]
        else:
            x=self.proj_enc(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        cls_tokens = cls_tokens+pos_embed[:,:self.series_embed_len]

        if self.use_cls_token:
            x=torch.cat([cls_tokens,x],dim=1)

        #encoder_out=self.encoder(x_start)
        encoder_out=self.encoder(x.transpose(0,1)).transpose(0,1)
        #return encoder_out+x_start
        pool_kernel_size=self.input_len+self.series_embed_len if self.use_cls_token else self.input_len
        encoder_out_pooling=torch.nn.functional.avg_pool1d(encoder_out.transpose(1,2),kernel_size=pool_kernel_size).transpose(1,2)

        return encoder_out,encoder_out_pooling

    def forward_Pdecoder(self,x,x_pooling):#只使用线性层预测，不再使用transformer decoder
        N=x.shape[0] # batch, length(num_patches), dim
        #x=self.proj_dec(x.permute(0,2,1).to(torch.float32)).transpose(1,2)

        '''if self.use_else_tokens:
            x=x
        else:
            x=x[:,:self.series_embed_len,:]
            
        pred=x'''
        if self.use_else_tokens:
            pred=x
        else:
            pred=x[:,:self.series_embed_len,:] if self.use_cls_token else x_pooling

        #flatten        
        pred=pred.reshape(pred.shape[0],pred.shape[1]*pred.shape[2])

        #pred=self.decoder_pred(pred.permute(0,2,1).to(torch.float32)).transpose(1,2)
        pred=self.decoder_pred(pred)
        
        
        #print(pred)
        #pred=self.sigmoid(pred)
        #print(pred)
        return pred
    
    def forward_loss(self,pred,y):
        x=pred.squeeze(-1).float()
        y=y.long()
        loss_func=nn.CrossEntropyLoss()
        #print(x.shape)
        #print(y.shape)
        loss = loss_func(x,y).float()
        return x,loss
    
    def forward(self,x,y):
        encoder_out,encoder_out_pooling=self.forward_encoder(x)
        pred=self.forward_Pdecoder(encoder_out,encoder_out_pooling)
        pred,loss=self.forward_loss(pred=pred,y=y)
        return loss,pred
