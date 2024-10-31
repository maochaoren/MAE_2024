import torch 
import os
from data_loader import Dataset_UCR,Dataset_epilepsy,Dataset_HAR
from torch.utils.data import DataLoader
from mae_model import MaskedAutoEncoder as mae_model
from MAE_BlockMask import BlockMaskedAutoEncoder as time_block_mae
from RetroMAElike_model import RetroMaskedAutoEncoder as retro_mae
from torchtools import EarlyStopping
#torch.backends.cudnn.enable =True
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

'''def get_device():
    return 'cpu'''


#预训练：解码器应该较弱，迫使编码器学习的更好。

#参数
patience=4
train_shuffle=True
base_lr=1e-3
epochs=8

mask_rate=0.75
encoder='trans'
train_pe=False
encoder_depth=3
decoder_depth=1
d_model=32
mask_size=1

#retro_mae
is_retro_mae=True
#mask_rate_enc=0.25
#mask_rate_dec=0.5
series_embed_len=1
enhance_decoding_list=[True]
mr_tuple_list=[[0.25,0.75]]
alpha=2
#mr_tuple_list=[[0.25,0.9],[0.5,0.9],[0.75,0.9],[0.25,0.5],[0.25,0.75],[0.5,0.5],[0.5,0.75]]

#数据集、预训练模型
data_set='HAR'
#data_set='Epilepsy'
model='retro_mae' if is_retro_mae else 'mae'
#model='mae'
#model='time_block_mae'

if data_set=='UCR':
    Dataset=Dataset_UCR
    data_list=os.listdir('UCRArchive_2018/')
    data_name='UCR'
if data_set=='Epilepsy':
    Dataset=Dataset_epilepsy
    data_list=['epilepsy']
    data_name='epilepsy'
if data_set=='HAR':
    Dataset=Dataset_HAR
    data_list=['HAR']
    data_name='HAR'


for data_path in data_list:
    dataset_train=Dataset(data_name=str(data_path),scale=True,flag='train')
    if data_name=='UCR':
        dataset_val=Dataset(data_name=str(data_path),scale=True,flag='test')
    else:
        dataset_val=Dataset(data_name=str(data_path),scale=True,flag='val')
    #print(dataset_train.input_len())
    data_loader_train=DataLoader(dataset=dataset_train,batch_size=8,shuffle=train_shuffle,drop_last=False)
    data_loader_val=DataLoader(dataset=dataset_val,batch_size=8,shuffle=False,drop_last=False)
    #data_path='Worms'
    for enhance_decoding in enhance_decoding_list:
        for mr_tuple in mr_tuple_list:
            mask_rate_enc=mr_tuple[0]
            mask_rate_dec=mr_tuple[1]

                #data_path='Fungi'

            if model=='mae':
                mae=mae_model(c_in=dataset_train.c_in(),d_model=d_model,input_len=dataset_train.input_len(),mask_rate=mask_rate,encoder_depth=encoder_depth,decoder_depth=decoder_depth,mask_size=mask_size,freq='h',train_pe=train_pe,encoder=encoder).to(get_device())
                model_save_path='pre_train_mae_'+data_name+'/'
                #print(mae.Transformer_Encoder)
            elif model=='time_block_mae':
                mae=time_block_mae(c_in=dataset_train.c_in(),d_model=d_model,input_len=dataset_train.input_len(),mask_rate=mask_rate,encoder_depth=encoder_depth,decoder_depth=decoder_depth,time_block=mask_size).to(get_device())
                model_save_path='pre_train_TBmae_'+data_name+'/'
            elif model=='retro_mae':
                mae=retro_mae(c_in=dataset_train.c_in(),d_model=d_model,input_len=dataset_train.input_len(),series_embed_len=series_embed_len,mask_rate_enc=mask_rate_enc,mask_rate_dec=mask_rate_dec,encoder_depth=encoder_depth,decoder_depth=decoder_depth,mask_size=1,enhance_decoding=enhance_decoding,freq='h',
                              alpha=alpha,train_pe=train_pe).to(get_device())
                model_save_path='pre_train_Retro_mae_'+data_name+'/'
            #optimizer=torch.optim.AdamW(mae.parameters(),lr=base_lr,betas=(0.9,0.95),weight_decay=0.05)
            optimizer=torch.optim.Adam(mae.parameters(),lr=base_lr,betas=(0.9,0.999))
            if model!='retro_mae':
                early_stopping=EarlyStopping(patience=patience,verbose=True,path=model_save_path+str(data_path)+'_'+str(mask_size)+'_'+str(mask_rate)+'.pt')
            else:
                early_stopping=EarlyStopping(patience=patience,verbose=True,path=model_save_path+str(data_path)+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt')
            
            print(data_path)
            for epoch in range(0,epochs):
                print('epoch: {}/{}'.format(epoch+1,epochs))
                #训练
                total_train_loss=0
                for data in data_loader_train:# 
                    mae.train()
                    x,_=data
                    if x.ndim==2:
                        x=x.unsqueeze(-1) 
                    x=x.float().to(get_device())
                    loss=mae(x)
                    total_train_loss+=loss*x.shape[0]
                    #print(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_train_loss/=len(data_loader_train.dataset)
                print("total_train_loss:{}".format(total_train_loss))
                #验证
                total_val_loss=0
                mae.eval() 
                for data in data_loader_val:
                    with torch.no_grad():
                        x,_=data
                        if x.ndim==2:
                            x=x.unsqueeze(-1) 
                        x=x.float().to(get_device())
                        loss=mae(x)
                        #print("val:{}".format(loss))
                        total_val_loss+=loss*x.shape[0]
                total_val_loss/=len(data_loader_val.dataset)
                print("total_val_loss:{}".format(total_val_loss))
            if model!='retro_mae':
                print('model={} data={} encoder_depth={} decoder_depth={} mask_rate={} lr={} d_model={} patch_size={} train_shuffle:{} \n min_val_loss:{}'
                .format(model,str(data_path),encoder_depth,decoder_depth,mask_rate,base_lr,d_model,mask_size,train_shuffle,early_stopping.val_loss_min))
            else:
                print('model={} enhance_decoding={} series_embed_len:{} data={} encoder_depth={} decoder_depth={} enc_mask_rate={} dec_mask_rate={} lr={} d_model={} patch_size={} train_shuffle:{} \n min_val_loss:{}'
                .format(model,enhance_decoding,series_embed_len,str(data_path),encoder_depth,decoder_depth,mask_rate_enc,mask_rate_dec,base_lr,d_model,mask_size,train_shuffle,early_stopping.val_loss_min))
            
            #保存
            model_path=model_save_path+str(data_path)+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt' if is_retro_mae else model_save_path+str(data_path)+'_'+str(mask_size)+'_'+str(mask_rate)+'.pt'
            torch.save(mae,model_path)

            if is_retro_mae==False:
                break   

        if is_retro_mae==False:
            break   
            #model_path='pre_train_model/model_ETTh2.pt'
            #torch.save(mae,model_path)

