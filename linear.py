import torch.nn as nn
import torch
from RetroMAElike_model import *
from MultiDecompMAE import series_decomp,MultiDecompEncoder

class decomp_Linear(nn.Module):
    def __init__(self, c_in,input_len,pred_len, window_size=24*7+1,CI=True,part='t'):
        super(decomp_Linear, self).__init__()
        self.CI=CI
        self.input_len = input_len
        self.pred_len = pred_len
        self.part=part
        self.linear = nn.Linear(input_len, pred_len, bias=True)
        self.series_decomp = series_decomp(kernel_size=window_size)
        self.mse_loss=nn.MSELoss()
        self.revin_layer=RevIN(c_in,affine=True)

    def dim_loss(self,pred,pred_label,dim):
        x=pred[:,:,dim]
        pred_label=pred_label[:,:,dim].float()
        loss_func=nn.MSELoss()
        loss = loss_func(x,pred_label).float()
        return loss

    def forward(self, x,x_pred):
        n_dim=x.shape[2]
        x_s,x_t=self.series_decomp(x)

        x_s_pred,x_t_pred=self.series_decomp(x_pred)

        x_label=x_t_pred if self.part=='t' else x_s_pred

        x_t,means,stdev=instance_norm(x_t)
        #x_t=x_t.permute(0,2,1)
        #print(x_t.shape)
        #x_t=self.revin_layer(x_t,'norm')
        #x_t=x_t.permute(0,2,1)

        if self.CI:
            x=x.permute(0,2,1)
            x=torch.reshape(x,(x.shape[0]*x.shape[1],x.shape[2])).unsqueeze(-1)
        
        x = self.linear(x.permute(0,2,1).float()).transpose(1,2)
        #print(x.shape)
        if self.CI:
            x=torch.reshape(x,(x.shape[0]//n_dim,n_dim,x.shape[1]))
            x=x.permute(0,2,1)
        
        if self.part=='t':
            #x=x.permute(0,2,1)
            #print(x.shape)
            #x=self.revin_layer(x,'denorm')
            x=instance_denorm(x,means,stdev,self.pred_len)
            #x=x.permute(0,2,1)
        loss=self.mse_loss(x,x_label)
            
        return loss,x

