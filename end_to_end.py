import torch
import torch.nn as nn
from data_loader import Dataset_ETT_hour,Dataset_ETT_minute,Dataset_ECL,Dataset_ETT_hour_disentangle
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.forecasting import stl
from torchtools import EarlyStopping
from Embedding import PatchEmbedding,PositionalEmbedding
from RetroMAElike_model import instance_denorm,instance_norm

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class LinearPred(nn.Module):
    def __init__(self,input_len,pred_len,is_norm):
        super(LinearPred,self).__init__()
        self.pred_len=pred_len
        self.pred=nn.Linear(input_len,pred_len)
        self.loss=nn.MSELoss()
        self.is_norm=is_norm
        
    def forward(self,x,x_pred):
        if self.is_norm:
            x,means,stdev=instance_norm(x)
            x=self.pred(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
            x=instance_denorm(x,means,stdev,self.pred_len)
        else:
            x=self.pred(x.permute(0,2,1).to(torch.float32)).transpose(1,2)
        loss=self.loss(x,x_pred)
        return loss,x
    
    def dim_loss(self,pred,pred_label,dim):
        x=pred[:,:,dim]
        pred_label=pred_label[:,:,dim].float()
        loss_func=nn.MSELoss()
        loss = loss_func(x,pred_label).float()
        return loss
    
    
if __name__=='__main__':

    data_set='hour1'

    input_len=96
    #pred_len=192
    pred_len_list=[720]
    model_name='linear'
    is_norm=False
    mode='t' # s/t/both
    todo='train' #train/test
    is_plt=True
    
    epochs=10
    lr=1e-3

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
    
    
    img_path='img_t/'
    loss_func=nn.MSELoss()
    
    for pred_len in pred_len_list:

        dataset_train=Dataset(root_path=root_path,flag='train',features='M',size=[input_len,input_len,pred_len],scale=False,mode=mode)
        data_loader_train=DataLoader(dataset=dataset_train,batch_size=1,shuffle=True,drop_last=False)
        dataset_val=Dataset(root_path=root_path,flag='val',features='M',size=[input_len,input_len,pred_len],scale=False,mode=mode)
        data_loader_val=DataLoader(dataset=dataset_train,batch_size=1,shuffle=False,drop_last=False)
        
        if todo=='train':

            if model_name=='linear':
                model=LinearPred(input_len=input_len,pred_len=pred_len,is_norm=is_norm).to(get_device())
            #model=Distangle(input_dims=dataset_train.dim(),output_dims=32,kernels=[1, 2, 4, 8, 16, 32, 64, 128],length=input_len)
            optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999))
            early_stopping=EarlyStopping(patience=2,verbose=True,path='end_to_end_model/'+data_name+'_'+mode+'_'+str(input_len)+'_'+str(pred_len)+'.pt',delta=5e-4) 

            for epoch in range(0,epochs):
                model.train()
                total_train_loss=0

                for step,(data) in enumerate(data_loader_train):
                    '''y,_,_,_=data

                if step%input_len==0:
                    x=np.arange(input_len)
                    for dim_show in range(0,dataset_train.dim()):
                        y_label=y[:,:,dim_show].squeeze(0).numpy()
                        plt.plot(x,y_label,color='b',linewidth=1)
                        title=data_name+' week:'+str(int(step/input_len+1))+' dim:'+str(dim_show)
                        plt.title(title)
                        path=img_path+str(int(step/input_len+1))+'_'+str(dim_show)+'.jpg'
                        plt.savefig(path)
                        plt.close()'''
                    x,y=data
                    x=x.float()
                    y=y.float()
                    x=x.to(get_device())
                    y=y.to(get_device())
                    x_pred=y[:,input_len:,:]
                    loss,pred=model(x,x_pred)
                    total_train_loss+=loss*x.shape[0]

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_train_loss/=len(data_loader_train.dataset)

                total_val_loss=0
                model.eval()
                for data in data_loader_val:
                    with torch.no_grad():
                        x,y=data
                        x=x.float()
                        y=y.float()
                        x=x.to(get_device())
                        y=y.to(get_device())
                        x_pred=y[:,input_len:,:]
                        loss,pred=model(x,x_pred)
                        total_val_loss+=loss*x.shape[0]

                total_val_loss/=len(data_loader_val.dataset)

                print('train_loss:{}'.format(total_train_loss))
                print('val_loss:{}'.format(total_val_loss))

                early_stopping(total_val_loss,model)
                if early_stopping.early_stop:
                    break
                
        if todo=='test':
            model=torch.load('end_to_end_model/'+data_name+'_'+mode+'_'+str(input_len)+'_'+str(pred_len)+'.pt')
            dataset_test=Dataset(root_path=root_path,flag='test',features='M',size=[input_len,input_len,pred_len],scale=False,mode=mode)
            data_loader_test=DataLoader(dataset=dataset_test,batch_size=1,shuffle=False,drop_last=False)

            total_test_loss=0
            dim_val_loss=np.zeros(dataset_test.dim())
            model.eval()
            for step,(data) in enumerate(data_loader_test):
                with torch.no_grad():
                    x,y=data
                    x=x.float()
                    y=y.float()
                    x=x.to(get_device())
                    y=y.to(get_device())
                    x_pred=y[:,input_len:,:]
                    loss,pred=model(x,x_pred)
                    print(loss)
                    total_test_loss+=loss*x.shape[0]

                    for dim_show in range(0,dataset_test.dim()):
                            dim_val_loss[dim_show]+=loss_func(pred[:,:,dim_show],x_pred[:,:,dim_show])

                    if is_plt:
                        if step%pred_len==0:
                            y=y[:,input_len:,:]
                            for dim_show in range(0,dataset_test.dim()):
                                x=np.arange(pred_len)
                                y_pred=pred[:,:,dim_show].squeeze(0).cpu().numpy()
                                y_label=y[:,:,dim_show].squeeze(0).cpu().numpy()
                                dim_loss=loss_func(pred[:,:,dim_show],x_pred[:,:,dim_show])
                                plt.plot(x,y_pred,color='red',linewidth=1)
                                plt.plot(x,y_label,color='blue',linewidth=1)
                                title='dim:'+str(dim_show)+' dim_loss:'+str(float(dim_loss))+' total_loss:'+str(float(loss))
                                plt.title(title)
                                path=img_path+str(int(step/pred_len))+'_'+str(dim_show)+'.jpg'
                                plt.savefig(path)
                                plt.close()
            total_test_loss/=len(data_loader_test.dataset)
            dim_val_loss/=len(data_loader_test.dataset)
            print('total_test_loss:{}'.format(total_test_loss))
            for dim in range(0,dataset_test.dim()):
                print('dim{}: total_test_loss:{}'.format(dim,dim_val_loss[dim]))
