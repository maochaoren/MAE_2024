import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data_loader import *
from torch.utils.data import DataLoader
from MultiDecompMAE import series_decomp
from torchtools import EarlyStopping
from fine_tuning_model_new import FineTuningModelNew
from fine_tuning_model_distangle import FineTuningModelDistangle
from RetroMAElike_model import instance_denorm,instance_norm
from SimMTM import topk_fft_decomp
import pyspark

def mask_func(x_s,mask_ratio):
    N, L, D = x_s.shape # batch, length(num_patches), dim
    len_keep = int(L * (1 - mask_ratio))
    mask_rand=torch.rand(N,L)
    ids_shuffle = torch.argsort(mask_rand, dim=1)  # ascend: small is keep, large is remove(和mask_ratio相比) 其实是第0维（同一行）之间排序
    ids_keep_restore = torch.sort(ids_shuffle[:,:len_keep], dim=1).values
    ids_restore=torch.argsort(torch.cat([ids_keep_restore,ids_shuffle[:,len_keep:]],dim=1),dim=1)#不改变保留token的相对次序
    #小的留下，大的mask掉
    #print(x.shape)
    x_s_masked = torch.gather(x_s, dim=1, index=ids_keep_restore.unsqueeze(-1).repeat(1, 1, D).long())#gather:按照给定的idx挑选第dim维保留的元素，然后将第dim维消去。
    #print(x_masked.shape)
    mask = torch.ones([N, L])
    mask[:, :len_keep] = 0 #在mask数组中，0表示未被Mask，1表示被Mask的
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    mask_tokens=nn.Parameter(torch.zeros(ids_restore.shape[0],int(L*mask_ratio),D),requires_grad=False)
    x_s_masked=torch.concat([x_s_masked,mask_tokens],dim=1)
    x_s_masked =torch.gather(x_s_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D).long())
    return x_s_masked


class fft_decomp(nn.Module):
    def __init__(self,st_sep,dom_season,lpf,topk=500):
        super(fft_decomp,self).__init__()
        self.st_sep=st_sep
        self.dom_season=dom_season
        self.lpf=lpf
        self.topk=topk
    def convert_coeff(self, x, eps=1e-6):#从复数得到相位和振幅
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))#real：实部；imag：虚部。
        phase = torch.atan2(x.imag, x.real + eps)#atan2:求向量（张量）和x轴的夹角（等同tan^-1)
        return amp, phase
    def forward(self,x):
        _,L,_=x.shape
        padding=nn.Parameter(torch.zeros(x.shape[0],L*(10-1),x.shape[2]),requires_grad=False)
        x=torch.cat([x,padding],dim=1)

        x_fft=torch.fft.rfft(x,dim=1)
        #if self.lpf>0:
            #x_fft[:,(x.shape[1]//self.dom_season)*self.lpf:,:]=0
        x_s=x_fft.clone()
        x_t=x_fft.clone()
        
        #sep=int((x.shape[1]//self.dom_season)*self.st_sep)#主波频率的倍数
        sep=int(st_sep*10)
        #sep
        x_s[:,:sep,:]=0
        #x_s[:,150:,:]=0
        x_t=x_fft-x_s
        
        '''amp=self.convert_coeff(x_s)[0]
        topk_values,topk_indices=torch.topk(amp,k=self.topk,dim=1)
        mask=torch.zeros_like(x_s).float()
        mask.scatter_(1,topk_indices,torch.ones_like(topk_indices).float())
        x_s=x_s*mask'''

        x_s=torch.fft.irfft(x_s,dim=1)[:,:L,:]
        x_t=torch.fft.irfft(x_t,dim=1)[:,:L,:]
        x_fft=self.convert_coeff(x_fft)[0]

        return x_fft,x_s,x_t
    
    def convert_coeff(self, x, eps=1e-6):#从复数得到相位和振幅
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))#real：实部；imag：虚部。
        phase = torch.atan2(x.imag, x.real + eps)#atan2:求向量（张量）和x轴的夹角（等同tan^-1)
        return amp, phase

dataset_dict={0:'hour1',1:'hour2',2:'ECL',3:'min1',
              4:'min2',5:'sim1',6:'sim2'}
data_set=dataset_dict[3]

if data_set=='hour2':
    Dataset=Dataset_ETT_hour
    freq='h'
    data_path='ETTh2.csv'
    data_name='ETTh2'
    root_path='ETT'
elif data_set=='hour1':
    Dataset=Dataset_ETT_hour
    freq='h'
    data_path='ETTh1.csv'
    data_name='ETTh1'
    root_path='ETT'
elif data_set=='min1':
    Dataset=Dataset_ETT_minute
    freq='t'
    data_path='ETTm1.csv'
    data_name='ETTm1'
    root_path='ETT'
elif data_set=='min2':
    Dataset=Dataset_ETT_minute
    freq='t'
    data_path='ETTm2.csv'
    data_name='ETTm2'
    root_path='ETT'
elif data_set=='ECL':
    Dataset=Dataset_ECL
    freq='h'
    data_path='ECL.csv'
    data_name='ECL'
    root_path='ECL'
elif data_set=='sim1':
    Dataset=Dataset_Sim
    freq='h'
    data_path='sim_data1.csv'
    data_name='sim1'
    root_path='sim_dataset'
elif data_set=='sim2':
    Dataset=Dataset_Sim
    freq='h'
    data_path='sim_data2.csv'
    data_name='sim2'
    root_path='sim_dataset'

input_len=336
dom_freq=96 if data_set=='min1' or data_set=='min2' else 24
window_size=24+1
st_sep=1.5
decomp='fft'
top_k_freq=5
daset_train=Dataset(root_path=root_path,data_path=data_path,flag='test',features='M',size=[input_len,input_len,1],scale=False if data_set=='sim2' else True)
data_loader_train=DataLoader(daset_train,batch_size=1,shuffle=False,num_workers=0)
img_path='img/'+decomp+'/'
dim_list=[0,1,2,3,4,5,6]
show_len=input_len
fft_decomp=fft_decomp(st_sep=st_sep,dom_season=dom_freq,lpf=0)
topk_fft_decomp=topk_fft_decomp(k=top_k_freq)
series_decomp=series_decomp(kernel_size=window_size)
for step,(data) in enumerate(data_loader_train):
    if step%input_len==0:
        x,_,_,_=data
        x=instance_norm(x)[0]
        x_raw=x
        if decomp=='fft':
            #x_mask=mask_func(x,0.5)
            x_fft_,x_s,x_t=fft_decomp(x)
            x_fft=torch.fft.rfft(x,dim=1)
            #x_masked_fft=torch.fft.rfft(x_mask,dim=1)
            x_amp,x_phase=fft_decomp.convert_coeff(x_fft)
            #x_amp_masked,x_phase_masked=fft_decomp.convert_coeff(x_masked_fft)

            #x_amp_topk=torch.topk(x_amp,k=top_k_freq,dim=1)[1].squeeze(0)
            _,x_s,x_t=fft_decomp(x)
            #x_s,_=topk_fft_decomp(x_s)
            #print(x_amp_topk.shape)
        else:
            x_s,x_t=series_decomp(x)
        #x_t=x_fft-x_s

        index=np.arange(0,show_len,1)
        for dim_show in dim_list:
            #print(x_amp)
            #print(x.shape)
            #print(x_t)
            y_raw=x[:,:show_len,dim_show].squeeze(0).numpy()
            y_s=x_s[:,:show_len,dim_show].squeeze(0).numpy()
            y_t=x_t[:,:show_len,dim_show].squeeze(0).numpy()
            #y_fft=x_fft[:,:show_len,dim_show].squeeze(0).numpy()
            #y_fft_=x_fft_[:,:show_len,dim_show].squeeze(0).numpy()
            #y_amp=x_amp[:,:show_len,dim_show].squeeze(0).numpy()
            #y_phase=x_phase[:,:show_len,dim_show].squeeze(0).numpy()
            #y_amp_masked=x_amp_masked[:,:show_len,dim_show].squeeze(0).numpy()
            #y_phase_masked=x_phase_masked[:,:show_len,dim_show].squeeze(0).numpy()
            plt.plot(index,y_raw,color='green',linewidth=1)
            plt.plot(index,y_s,color='red',linewidth=1)
            plt.plot(index,y_t,color='blue',linewidth=1)
            #plt.plot(y_fft_,color='purple',linewidth=1)

            #plt.plot(index,y_amp,color='green',linewidth=1)
            #plt.plot(index,y_amp_masked,color='red',linewidth=1)


            #plt.scatter(index,y_phase,color='red',linewidth=0.5) 
            #plt.scatter(index,y_phase_masked,color='blue',linewidth=0.5) 
            path=img_path+str(int(step/input_len))+'_'+str(dim_show)+'.jpg'
            plt.savefig(path)
            plt.close()

