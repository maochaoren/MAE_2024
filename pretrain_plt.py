import torch 
from data_loader import Dataset_ETT_hour,Dataset_ECL,Dataset_ETT_minute
from torch.utils.data import DataLoader
from torchtools import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'



input_len=96
pred_len_list=[96]
data_set='hour1'
is_retro_mae=True
#data_set='minute'
mask_rate=0.75
mask_size=1


model='retro_mae' if is_retro_mae else 'mae'
enhance_decoding=True
mask_rate_enc=0.25
mask_rate_dec=0.75

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
elif data_set=='minute':
    Dataset=Dataset_ETT_minute
    freq='t'
    data_path='ETTm1.csv'
    data_name='ETTm1'
    root_path='ETT'
elif data_set=='ECL':
    Dataset=Dataset_ECL
    freq='h'
    data_path='ECL.csv'
    data_name='ECL'
    root_path='ECL'

if model=='mae':
    model_path='pre_train_mae/'
elif model=='time_block_mae':
    model_path='pre_train_BTmae/'
elif model=='retro_mae':
    model_path='pre_train_Retro_mae/'
elif model=='multidec_mae':
    model_path='pre_train_multidec_mae/'

for pred_len in pred_len_list:
    if model=='retro_mae':
        path=model_path+data_name+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
        img_path='img_train/retro_mae/'+data_name+'/'
    else:
        path=model_path+data_name+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(mask_rate)+'.pt'
        img_path='img_train/mae/'+data_name+'/'

    size_tuple=[input_len,input_len,pred_len]


    dataset=Dataset(root_path=root_path,data_path=data_path,flag='val',features='M',size=size_tuple,scale=True)
    data_loader_train=DataLoader(dataset=dataset,batch_size=1,shuffle=False,drop_last=False)
    model=torch.load(path)
    model.eval()
    total_val_loss=0
    dim_val_loss=np.zeros(dataset.dim())

    #分预测
    for step,(data) in enumerate(data_loader_train):
        with torch.no_grad():
            x,y,x_mask,y_mask=data
            x=x.to(get_device())
            x_mask=x_mask.to(get_device())
            y=y.to(get_device())
            y_mask=y_mask.to(get_device())
            x_label=y[:,:size_tuple[1],:]
            x_label_mask=y_mask[:,:size_tuple[1],:]
            x_pred=y[:,size_tuple[1]:,:]
            x_pred_mask=y_mask[:,size_tuple[1]:,:]
            pred,loss=model(x)
            
            total_val_loss+=loss*x.shape[0]
            for dim_show in range(0,dataset.dim()):
                dim_val_loss[dim_show]+=model.dim_loss(x,pred,dim_show)*x.shape[0]
        #绘图 预测红色 原数据蓝色
        if step%pred_len==0:
            y=x#y:原始输入

            for dim_show in range(0,dataset.dim()):
                x=np.arange(input_len)
                y_pred=pred[:,:,dim_show].squeeze(0).cpu().numpy()
                y_label=y[:,:,dim_show].squeeze(0).cpu().numpy()
                dim_loss=model.dim_loss(y,pred,dim_show)

                for i in range(0,input_len):
                    if model.mask[:,i]==1:
                        plt.scatter(x[i],y_pred[i],color='red',s=8)
                plt.plot(x,y_label,color='blue',linewidth=1)
                plt.plot(x,y_pred,color='red',linewidth=0.5)
                #print(dim_loss)
                #print(loss.shape)
                title='dim:'+str(dim_show)+' dim_loss:'+str(float(dim_loss))+' total_loss:'+str(float(loss))
                plt.title(title)
                path=img_path+str(int(step/pred_len))+'_'+str(dim_show)+'.jpg'
                plt.savefig(path)
                plt.close()

            #print(y_pred.shape)


    total_val_loss/=len(data_loader_train.dataset)
    dim_val_loss/=len(data_loader_train.dataset)

    print('total_test_loss:{}'.format(total_val_loss))
    for dim in range(0,dataset.dim()):
        print('dim{}: total_test_loss:{}'.format(dim,dim_val_loss[dim]))


