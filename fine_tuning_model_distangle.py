import torch.nn as nn
import torch
from Embedding import PatchEmbedding,PositionalEmbedding

class FineTuningModelDistangle(nn.Module):
    def __init__(self,c_in,d_model,input_len,series_embed_len,distangle_mae_encoder,mask_size,pred_len,use_cls_token=False
        ):
        super(FineTuningModelDistangle, self).__init__()
        self.PositionalEmbedding=PositionalEmbedding(d_model=d_model)
        #encoder
        self.conv_embed=distangle_mae_encoder.conv_embed
        self.distangle=distangle_mae_encoder.distangle
        self.encoder_s=distangle_mae_encoder.season_mae
        self.encoder_t=distangle_mae_encoder.trend_mae
        #decoder
        self.proj_dec=nn.Conv1d(d_model,d_model,kernel_size=3,padding=1,padding_mode='circular')
        self.decoder_proj=nn.Linear(d_model//2,c_in,bias=True)
        #self.decoder_proj=nn.Linear(d_model,c_in,bias=True)
        self.decoder_pred=nn.Linear(input_len+series_embed_len,pred_len) if use_cls_token else nn.Linear(input_len,pred_len)
        
    def forward_pred(self,x):#只使用线性层预测，不再使用transformer decoder
        pred=self.decoder_proj(x)
        pred=self.decoder_pred(pred.permute(0,2,1).to(torch.float32)).transpose(1,2)

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
        #x=x.float()
        x=self.conv_embed(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
        x_trend,x_season=self.distangle(x)#embedding,distangle
        #x_season=x_season.float()
        #x_trend=x_trend.float()
        encoder_out_t=self.encoder_t.repr_gen(x_trend,pe=True,fine_tune=True)
        encoder_out_s=self.encoder_s.repr_gen(x_season,pe=True,fine_tune=True)
        
        #repr=torch.concat([encoder_out_t,encoder_out_s],dim=2)
        repr=encoder_out_s+encoder_out_t
        pred=self.forward_pred(repr)

        loss=self.forward_loss(pred=pred,pred_label=x_pred)

        return loss,pred
