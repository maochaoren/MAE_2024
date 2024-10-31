import torch 
from data_loader import Dataset_ETT_hour,Dataset_ETT_minute,Dataset_ECL,Dataset_ETT_hour_disentangle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from end_to_end import LinearPred

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

data_set='hour1'


input_len=96
pred_len_list=[96,192,336,720]
mode='both'
model_t='linear' #mae / linear
mode_list=['s','t']
#mode_list=['t']
flag='test' #tain/val/test
is_retro_mae=True
#data_set='minute'
mask_rate=0.75
mask_size=1

model='retro_mae' if is_retro_mae else 'mae'
#model='mae'
#model='distangle_mae'

enhance_decoding=True
mask_rate_enc=0.5
mask_rate_dec=0.75

#是否画图
is_plt=True
#plot_list=[1,2,5]
plot_list=[0,1,2,3,4,5,6]

criterion=torch.nn.MSELoss()

if data_set=='hour2':
    Dataset=Dataset_ETT_hour_disentangle
    freq='h'
    data_name='ETTh2'
    root_path='ETTh2_disentangle'
elif data_set=='hour1':
    Dataset=Dataset_ETT_hour_disentangle
    freq='h'
    data_name='ETTh1'
    root_path='ETTh1_disentangle'

for pred_len in pred_len_list:
    print('pred_len:{}'.format(pred_len))
    size_tuple=[input_len,input_len,pred_len]
    #分预测
    if mode=='s' or mode=='t':
        for mode in mode_list:
            if model=='mae':
                mae_encoder_path='pre_train_mae/'+data_name+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(mask_rate)+'.pt'
                ft_path='fine_tuning_mae_st/'
            elif model=='time_block_mae':
                mae_encoder_path='pre_train_TBmae/'+data_name+'_'+str(input_len)+'_'+str(mask_size)+'.pt'
                ft_path='fine_tuning_BTmae/'
            elif model=='retro_mae':
                mae_encoder_path='pre_train_Retro_mae_'+str(mode)+'/'+data_name+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
                ft_path='fine_tuning_retro_mae_st/'

            img_path='img/'+model+'_'+mode+'/'

            dataset=Dataset(root_path=root_path,flag=flag,features='M',size=size_tuple,scale=False,mode=mode)
            data_loader_valid=DataLoader(dataset=dataset,batch_size=1,shuffle=False,drop_last=True)

            path=ft_path+'ft_'+data_name+'_'+mode+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'     
            if model_t=='linear' and mode=='t':
                path='end_to_end_model/'+data_name+'_'+'t'+'_'+str(input_len)+'_'+str(pred_len)+'.pt'

            ft_model=torch.load(path) 

            ft_model.eval()
            total_val_loss=0
            dim_val_loss=np.zeros(dataset.dim())

            for step,(data) in enumerate(data_loader_valid):
                with torch.no_grad():
                    x,y=data
                    x=x.to(get_device())
                    y=y.to(get_device())
                    x_pred=y[:,size_tuple[1]:,:]
                    loss,pred=ft_model(x,x_pred)
                    '''pred=dataset.inverse_transform(pred)
                    x_pred=dataset.inverse_transform(x_pred)
                    criterion=torch.nn.MSELoss()
                    loss=criterion(pred,x_pred)'''
                    #print("batch{}:val_loss={}".format(step,loss))
                    #print('eval_loss:{}'.format(loss))
                    total_val_loss+=loss*x.shape[0]
                    for dim_show in range(0,dataset.dim()):
                        dim_val_loss[dim_show]+=ft_model.dim_loss(pred,x_pred,dim_show)*x.shape[0]
                #绘图 预测红色 原数据蓝色
                if is_plt:
                    if step%pred_len==0:
                        y=y[:,size_tuple[1]:,:]

                        #for dim_show in range(0,dataset.dim()):
                        for dim_show in plot_list:
                            x=np.arange(pred_len)
                            y_pred=pred[:,:,dim_show].squeeze(0).cpu().numpy()
                            y_label=y[:,:,dim_show].squeeze(0).cpu().numpy()
                            dim_loss=ft_model.dim_loss(pred,x_pred,dim_show)
                            plt.plot(x,y_pred,color='red',linewidth=1)
                            plt.plot(x,y_label,color='blue',linewidth=1)
                            title='dim:'+str(dim_show)+' dim_loss:'+str(float(dim_loss))+' total_loss:'+str(float(loss))
                            plt.title(title)
                            path=img_path+str(int(step/pred_len))+'_'+str(dim_show)+'.jpg'
                            plt.savefig(path)
                            plt.close()

                        #print(y_pred.shape)


            total_val_loss/=len(data_loader_valid.dataset)
            dim_val_loss/=len(data_loader_valid.dataset)
            
            print('mode:{}'.format(mode))
            print('total_test_loss:{}'.format(total_val_loss))
            for dim in range(0,dataset.dim()):
                print('dim{}: total_test_loss:{}'.format(dim,dim_val_loss[dim]))
    elif mode=='both':#合预测
        if model=='retro_mae':
            ft_path='fine_tuning_retro_mae_st/'
            path_s=ft_path+'ft_'+data_name+'_'+'s'+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'     
        elif model=='mae':
            ft_path='fine_tuning_mae_st/'
            path_s=ft_path+'ft_'+data_name+'_'+'s'+'_'+str(input_len)+'_'+str(pred_len)+'.pt'

        if model_t=='mae':
            path_t=ft_path+'ft_'+data_name+'_'+'t'+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'  
        elif model_t=='linear':
            path_t='end_to_end_model/'+data_name+'_'+'t'+'_'+str(input_len)+'_'+str(pred_len)+'.pt'

        dataset=Dataset(root_path=root_path,flag=flag,features='M',size=size_tuple,scale=False,mode='both')
        data_loader_valid=DataLoader(dataset=dataset,batch_size=1,shuffle=False,drop_last=True)

        ft_model_s=torch.load(path_s)
        ft_model_t=torch.load(path_t)
        img_path='img/'+model+'_'+mode+'/'
        ft_model_s.eval()
        ft_model_t.eval()
        total_val_loss=0
        dim_val_loss=np.zeros(dataset.dim())

        for step,(data) in enumerate(data_loader_valid):
            with torch.no_grad():
                x_s,y_s,x_t,y_t=data
                x_s=x_s.to(get_device())
                y_s=y_s.to(get_device())
                x_t=x_t.to(get_device())
                y_t=y_t.to(get_device())
                x_pred_s=y_s[:,size_tuple[1]:,:]
                x_pred_t=y_t[:,size_tuple[1]:,:]

                _,pred_s=ft_model_s(x_s,x_pred_s)
                _,pred_t=ft_model_t(x_t,x_pred_t)

                pred=(pred_s+pred_t).float()
                loss=criterion(pred,x_pred_s+x_pred_t)
                total_val_loss+=loss*x_s.shape[0]
                for dim_show in range(0,dataset.dim()):
                    dim_val_loss[dim_show]+=ft_model_s.dim_loss(pred,x_pred_s+x_pred_t,dim_show)*x_s.shape[0]
                
                if is_plt:
                    if step%pred_len==0:
                        y=y_s[:,size_tuple[1]:,:]+y_t[:,size_tuple[1]:,:]

                        #for dim_show in range(0,dataset.dim()):
                        for dim_show in plot_list:
                            x=np.arange(pred_len)
                            y_pred=pred[:,:,dim_show].squeeze(0).cpu().numpy()
                            y_label=y[:,:,dim_show].squeeze(0).cpu().numpy()
                            dim_loss=ft_model_s.dim_loss(pred,x_pred_s+x_pred_t,dim_show)
                            plt.plot(x,y_pred,color='red',linewidth=1)
                            plt.plot(x,y_label,color='blue',linewidth=1)
                            title='dim:'+str(dim_show)+' dim_loss:'+str(float(dim_loss))+' total_loss:'+str(float(loss))
                            plt.title(title)
                            path=img_path+str(int(step/pred_len))+'_'+str(dim_show)+'.jpg'
                            plt.savefig(path)
                            plt.close()

                        #print(y_pred.shape)


        total_val_loss/=len(data_loader_valid.dataset)
        dim_val_loss/=len(data_loader_valid.dataset)
        print('mode:{}'.format(mode))
        print('total_test_loss:{}'.format(total_val_loss))
        for dim in range(0,dataset.dim()):
            print('dim{}: total_test_loss:{}'.format(dim,dim_val_loss[dim]))



