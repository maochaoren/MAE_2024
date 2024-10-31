import torch 
import os
from data_loader import Dataset_UCR,Dataset_epilepsy,Dataset_HAR
from torch.utils.data import DataLoader
from torchtools import EarlyStopping
from fine_tuning_model_CLS import FineTuningModelCLS

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
    #return 'cpu'
'''def get_device():
    return 'cpu'''

#不要使用transformer作为 fine_tune的decoder，只使用Linear即可。



#参数
is_retro_mae=True
frozen=False
frozen_num=6

train_pe=False
mask_size=1
d_model=32

base_lr=8e-5
epochs=100
patience=4
iters=1
series_embed_len=1
#data_set='minute'
data_set='HAR'
#data_set='Epilepsy'
#model='mae'
model='retro_mae' if is_retro_mae else 'mae'
enhance_decoding_list=[True]
#mr_tuple_list=[[0.25,0.9],[0.5,0.9],[0.75,0.9],[0.25,0.5],[0.25,0.75],[0.5,0.5],[0.5,0.75]]
mr_tuple_list=[[0.25,0.75]]
mask_rate=0.75

use_cls_token=True
use_else_tokens=False



if data_set=='UCR':
    Dataset=Dataset_UCR
    data_list=os.listdir('UCRArchive_2018/')
    data_name='UCR'
    data_list.sort(key=lambda x:x)
if data_set=='Epilepsy':
    Dataset=Dataset_epilepsy
    data_list=['epilepsy']
    data_name='epilepsy'
if data_set=='HAR':
    Dataset=Dataset_HAR
    data_list=['HAR']
    data_name='HAR'

for data_path in data_list:

    print(data_path)
    dataset_train=Dataset(data_name=str(data_path),flag='train',scale=True)
    data_loader_train=DataLoader(dataset=dataset_train,batch_size=32,shuffle=True,drop_last=False)

    if data_path!='UCR':
        dataset_val=Dataset(data_name=str(data_path),scale=True)
        data_loader_val=DataLoader(dataset=dataset_train,batch_size=32,shuffle=False,drop_last=False)

    for enhance_decoding in enhance_decoding_list:

        for mr_tuple in mr_tuple_list:
            mask_rate_enc=mr_tuple[0]
            mask_rate_dec=mr_tuple[1]
            avg_val_loss=0
            if model=='mae':
                mae_encoder_path='pre_train_mae_'+data_name+'/'+data_path+'_'+str(mask_size)+'_'+str(mask_rate)+'.pt'
                ft_path='fine_tuning_mae_'+data_name+'/'
                use_cls_token=False
            elif model=='time_block_mae':
                mae_encoder_path='pre_train_TBmae_'+data_name+'/'+data_path+'_'+str(mask_size)+'.pt'
                ft_path='fine_tuning_BTmae_'+data_name+'/'
                use_cls_token=False
            elif model=='retro_mae':
                mae_encoder_path='pre_train_Retro_mae_'+data_name+'/'+data_path+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
                ft_path='fine_tuning_Retro_mae_'+data_name+'/'
                
            mae=torch.load(mae_encoder_path)
            mae_encoder=mae.Transformer_Encoder
            mae_project_enc=mae.ScalarProjection_enc

            #冻结部分参数
            if frozen:
                count=0
                for blk in mae_encoder.layers:
                    if count==frozen_num:
                        break
                    for param in blk.parameters():
                        param.requires_grad=False
                    count+=1

                for param in mae_project_enc.parameters():
                    param.requires_grad=False

            ft_model=FineTuningModelCLS(c_in=1,d_model=d_model,input_len=dataset_train.input_len(),series_embed_len=series_embed_len,mae_encoder=mae_encoder,mae_project_enc=mae_project_enc,encoder_depth=3,mask_size=mask_size,
                            is_mae=True,freq='h',use_else_tokens=use_else_tokens,cls_num=dataset_train.cls_num(),train_pe=train_pe,use_cls_token=use_cls_token).to(get_device())
            #print(ft_model.is_mae)
            optimizer=torch.optim.Adam(ft_model.parameters(),lr=base_lr,betas=(0.9,0.999))
            #early_stopping=EarlyStopping(patience=4,verbose=True,path='ft_ETTm1_784_'+str(pred_len)+'.pt') 
            if model!='retro_mae':  
                early_stopping=EarlyStopping(patience=patience,verbose=True,path=ft_path+'ft_'+data_path+'_'+str(mask_rate)+'.pt',delta=5e-4) 
            else:
                early_stopping=EarlyStopping(patience=patience,verbose=True,path=ft_path+'ft_'+data_path+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt',delta=5e-4) 
                
            for iter in range(0,iters):#多次训练取平均
                for epoch in range(0,epochs):
                    #print('epoch: {}/{}'.format(epoch+1,epochs))
                    #训练
                    total_train_loss=0
                    total_train_correct=0
                    for step,(data) in enumerate(data_loader_train):

                        '''if step==0 and epoch==0:
                            print("data_length:{}".format(data[0].shape[1])) '''

                        ft_model.train()
                        x,y=data
                        x=x.long().to(get_device())
                        y=y.long().to(get_device())
                        #x=x.float()
                        loss,pred=ft_model(x,y)
                        total_train_loss+=loss*x.shape[0]
                        #print(pred)
                        #print(y)
                        total_train_correct+=torch.eq(pred.argmax(dim=1),y).sum().float()
                        #print(loss)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total_train_loss/=len(data_loader_train.dataset)
                    total_train_acc=total_train_correct/len(data_loader_train.dataset)
                    print("\ntotal_train_loss:{}".format(total_train_loss))
                    print("total_train_acc:{}".format(total_train_acc))
                    #验证
                    if data_name!='UCR':
                        total_val_loss=0
                        total_val_correct=0
                        ft_model.eval()
                        for data in data_loader_val:
                            with torch.no_grad():
                                x,y=data
                                x=x.long().to(get_device())
                                y=y.long().to(get_device())

                                loss,pred=ft_model(x,y)
                                total_val_loss+=loss*x.shape[0]
                                #print(pred)
                                #print(y)
                                total_val_correct+=torch.eq(pred.argmax(dim=1),y).sum().float()
                        total_val_loss/=len(data_loader_val.dataset)
                        total_val_acc=total_val_correct/len(data_loader_val.dataset)
                        print("total_val_loss:{}".format(total_val_loss))
                        print("total_val_acc:{}".format(total_val_acc))
                        #early stop
                        early_stopping(total_val_loss,ft_model)
                        if early_stopping.early_stop:
                            break

            if model != 'retro_mae':
                print('model={} data={} mask_size={} lr={}\n avg_val_loss:{}'
                  .format(model,data_path,mask_size,base_lr,avg_val_loss))
            else:
                print('model={} use_cls_token={} enhance_decoding={} mask_rate_enc={} mask_rate_dec={} data={} mask_size={} lr={}\n avg_val_loss:{}'
                  .format(model,use_cls_token,enhance_decoding,mask_rate_enc,mask_rate_dec,data_path,mask_size,base_lr,avg_val_loss))
            if model!='retro_mae':
                break
        if model!='retro_mae':
            break
    break
            