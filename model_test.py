import torch 
from data_loader import Dataset_ETT_hour,Dataset_ETT_minute,Dataset_ECL
from torch.utils.data import DataLoader
from torchtools import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'



input_len=336
pred_len_list=[96,192,336,720]
#pred_len_list=[96]

#是否画图
is_plt=True
#is_plt=True
#plot_list=[1,2,5]

data_set='min1'
print(data_set)
flag='test' #tain/val/test
is_retro_mae=True
#data_set='minute'
mask_rate=0.5

model='retro_mae' if is_retro_mae else 'mae'
#model='mae'
#model='retro_mae'
model='multidec_mae'
model='SimMTM'
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

if model=='mae':
    ft_path='fine_tuning_mae/'
elif model=='time_block_mae':
    ft_path='fine_tuning_BTmae/'
elif model=='retro_mae':
    ft_path='fine_tuning_Retromae/'
elif model=='distangle_mae':
    ft_path='fine_tuning_distangle_mae/'
elif model=='multidec_mae':
    ft_path='fine_tuning_multidec_mae/'
elif model=='SimMTM':
    ft_path='fine_tuning_SimMTM/'

for pred_len in pred_len_list:
    print('pred_len:{}'.format(pred_len))
    if model=='retro_mae':
        path=ft_path+'ft_'+data_name+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
        img_path='img/retro_mae/'+data_name+'/'
    elif model=='distangle_mae':
        path=ft_path+'ft_'+data_name+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
        img_path='img/distangle_mae/'+data_name+'/'
    elif model=='multidec_mae':
        path=ft_path+'ft_'+data_name+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(mask_rate)+'.pt'
        img_path='img/distangle_mae/'+data_name+'/'
    elif model=='SimMTM':
        path=ft_path+'ft_'+data_name+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(mask_rate)+'.pt'
        img_path='img/SimMTM/'+data_name+'/'
    else:
        path=ft_path+'ft_'+data_name+'_'+str(input_len)+'_'+str(pred_len)+'.pt'
        img_path='img/mae/'+data_name+'/'

    size_tuple=[input_len,input_len,pred_len]


    dataset=Dataset(root_path=root_path,data_path=data_path,flag=flag,features='M',size=size_tuple,scale=True)
    data_loader_valid=DataLoader(dataset=dataset,batch_size=1 if is_plt else 32,shuffle=False,drop_last=True)
    plot_list=np.arange(0,dataset.dim()).tolist()
    ft_model=torch.load(path)
    #print(ft_model)
    ft_model.eval()
    total_val_loss=0
    total_val_loss_s=0
    total_val_loss_t=0

    dim_val_loss=np.zeros(dataset.dim())

    for step,(data) in enumerate(data_loader_valid):
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
            _,pred,x_s,x_t,loss,loss_s,loss_t=ft_model(x,x_pred)
            #loss,pred=ft_model(x,x_pred)
            '''pred=dataset.inverse_transform(pred)
            x_pred=dataset.inverse_transform(x_pred)
            criterion=torch.nn.MSELoss()
            loss=criterion(pred,x_pred)'''
            #print("batch{}:val_loss={}".format(step,loss))
            #print('eval_loss:{}'.format(loss))
            total_val_loss+=loss*x.shape[0]
            total_val_loss_s+=loss_s*x.shape[0]
            total_val_loss_t+=loss_t*x.shape[0]
            for dim_show in range(0,dataset.dim()):
                dim_val_loss[dim_show]+=ft_model.dim_loss(pred,x_pred,dim_show)*x.shape[0]
            #if step==0:
            #    print(pred[:,:,0])
        #绘图 预测红色 原数据蓝色
        if is_plt:
            if step%pred_len==0:
                y=y[:,size_tuple[1]:,:]

                #for dim_show in range(0,dataset.dim()):
                for dim_show in plot_list:
                    x=np.arange(pred_len)
                    y_pred_t=x_t[:,:,dim_show].squeeze(0).cpu().numpy()
                    y_pred_s=x_s[:,:,dim_show].squeeze(0).cpu().numpy()
                    y_pred=pred[:,:,dim_show].squeeze(0).cpu().numpy()
                    y_label=y[:,:,dim_show].squeeze(0).cpu().numpy()
                    dim_loss=ft_model.dim_loss(pred,x_pred,dim_show)
                    plt.plot(x,y_pred_t,color='red',linewidth=1)
                    plt.plot(x,y_pred_s,color='blue',linewidth=1)
                    plt.plot(x,y_label,color='green',linewidth=2)
                    plt.plot(x,y_pred,color='purple',linewidth=2)
                    title='dim:'+str(dim_show)+' dim_loss:'+str(float(dim_loss))+' total_loss:'+str(float(loss))
                    plt.title(title)
                    path=img_path+str(int(step/pred_len))+'_'+str(dim_show)+'.jpg'
                    plt.savefig(path)
                    plt.close()

                #print(y_pred.shape)


    total_val_loss/=len(data_loader_valid.dataset)
    total_val_loss_s/=len(data_loader_valid.dataset)
    total_val_loss_t/=len(data_loader_valid.dataset)

    dim_val_loss/=len(data_loader_valid.dataset)

    print('total_test_loss__:{}'.format(total_val_loss))
    print('total_test_loss_s:{}'.format(total_val_loss_s))
    print('total_test_loss_t:{}'.format(total_val_loss_t))

    for dim in range(0,dataset.dim()):
        print('dim{}: total_test_loss:{}'.format(dim,dim_val_loss[dim]))


