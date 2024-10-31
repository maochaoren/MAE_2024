import torch 
import os
from data_loader import Dataset_ETT_hour_disentangle
from torch.utils.data import DataLoader
from mae_model import MaskedAutoEncoder as mae_model
from MAE_BlockMask import BlockMaskedAutoEncoder as time_block_mae
from RetroMAElike_model import RetroMaskedAutoEncoder as retro_mae
from torchtools import EarlyStopping
#torch.backends.cudnn.enable =True
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


#预训练：解码器应该较弱，迫使编码器学习的更好。
data_set='hour2'

#参数
pred_len=1
input_len_list=[96]
patience=3
train_shuffle=True
epochs=40
base_lr=1e-3

encoder='trans'
mask_rate=0.75
encoder_depth=3
decoder_depth=1
d_model=64
mask_size=1

#retro_mae
is_retro_mae=True
is_norm=False
#mask_rate_enc=0.25
#mask_rate_dec=0.5
series_embed_len=1
enhance_decoding_list=[True]
mr_tuple_list=[[0.5,0.75]]
alpha=0.5
#mr_tuple_list=[[0.25,0.9],[0.5,0.9],[0.75,0.9],[0.25,0.5],[0.25,0.75],[0.5,0.5],[0.5,0.75]]

#数据集、预训练模型
#data_set='minute'
mode_list=['s']

model='retro_mae' if is_retro_mae else 'mae'
#model='mae'
#model='retro_mae'
#model='distangle_mae'

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
'''elif data_set=='minute':
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
'''
for input_len in input_len_list:
    for enhance_decoding in enhance_decoding_list:
        for mr_tuple in mr_tuple_list:

            mask_rate_enc=mr_tuple[0]
            mask_rate_dec=mr_tuple[1]
                
            for mode in mode_list:
                dataset_train=Dataset(root_path=root_path,flag='train',features='M',size=[input_len,input_len,pred_len],scale=False,mode=mode)
                dataset_val=Dataset(root_path=root_path,flag='val',features='M',size=[input_len,input_len,pred_len],scale=False,mode=mode)
                #print(dataset_val.data_t.shape)
    
                data_loader_train=DataLoader(dataset=dataset_train,batch_size=64,shuffle=train_shuffle,drop_last=False)
                data_loader_valid=DataLoader(dataset=dataset_val,batch_size=64,shuffle=False,drop_last=False)
    
                if model=='mae':
                    mae=mae_model(c_in=dataset_train.dim(),d_model=d_model,input_len=input_len,mask_rate=mask_rate,encoder_depth=encoder_depth,decoder_depth=decoder_depth,mask_size=mask_size,freq=freq,encoder=encoder).to(get_device())
                    model_save_path='pre_train_mae_st/'
                elif model=='time_block_mae':
                    mae=time_block_mae(c_in=dataset_train.dim(),d_model=d_model,input_len=input_len,mask_rate=mask_rate,encoder_depth=encoder_depth,decoder_depth=decoder_depth,time_block=mask_size,freq=freq).to(get_device())
                    model_save_path='pre_train_TBmae/'
                elif model=='retro_mae':
                    mae=retro_mae(c_in=dataset_train.dim(),d_model=d_model,input_len=input_len,series_embed_len=series_embed_len,mask_rate_enc=mask_rate_enc,mask_rate_dec=mask_rate_dec,encoder_depth=encoder_depth,mask_size=1,enhance_decoding=enhance_decoding,alpha=alpha
                                  ,is_norm=is_norm).to(get_device())
                    model_save_path='pre_train_Retro_mae_st/'
                '''elif model=='distangle_mae':
                    mae=distangle_mae(c_in=dataset_train.dim(),d_model=d_model,input_len=input_len,mask_rate_enc=mask_rate_enc,mask_rate_dec=mask_rate_dec,encoder_depth=encoder_depth,enhance_decoding=enhance_decoding,alpha=alpha).to(get_device())
                    model_save_path='pre_train_distangle_mae/'''
                optimizer=torch.optim.Adam(mae.parameters(),lr=base_lr,betas=(0.9,0.999))
                if model=='retro_mae':
                    early_stopping=EarlyStopping(patience=patience,verbose=True,path=model_save_path+data_name+'_'+mode+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt')
                elif model=='mae':
                    early_stopping=EarlyStopping(patience=patience,verbose=True,path=model_save_path+data_name+'_'+mode+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(mask_rate)+'.pt')
                print(model)
                for epoch in range(0,epochs):
                    print('epoch: {}/{}'.format(epoch+1,epochs))
                    #训练
                    total_train_loss=0
                    for data in data_loader_train:# 四项：x,y,x_mask,y_mask
                        mae.train()
                        x,_=data
                        x=x.float()
                        x=x.to(get_device())
                        _,loss=mae(x)
                        #print(loss)
                        total_train_loss+=loss*x.shape[0]
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total_train_loss/=len(data_loader_train.dataset)
                    print("total_train_loss:{}".format(total_train_loss))
                    #验证
                    total_val_loss=0
                    mae.eval() 
                    for data in data_loader_valid:
                        with torch.no_grad():
                            x,_=data
                            x=x.float()
                            x=x.to(get_device())
                            _,loss=mae(x)
                            #print("val:{}".format(loss))
                            total_val_loss+=loss*x.shape[0]
                    total_val_loss/=len(data_loader_valid.dataset)
                    print("total_val_loss:{}".format(total_val_loss))
                    early_stopping(total_val_loss,mae)
                    if early_stopping.early_stop:
                        #print('input_len:{} best:{}'.format(input_len,early_stopping.val_loss_min))
                        print('early_stopping')
                        print('model:{}'.format(model))
                        if model=='retro_mae' or 'distangle_mae':
                            print('model={} enhance_decoding={} series_embed_len:{} data={} input_len={} encoder_depth={} decoder_depth={} enc_mask_rate={} dec_mask_rate={} lr={} d_model={} patch_size={} train_shuffle:{} \n min_val_loss:{}'
                            .format(model,enhance_decoding,series_embed_len,data_name,input_len,encoder_depth,decoder_depth,mask_rate_enc,mask_rate_dec,base_lr,d_model,mask_size,train_shuffle,early_stopping.val_loss_min))
                        else:
                            print('model={} data={} input_len={} encoder_depth={} decoder_depth={} mask_rate={} lr={} d_model={} patch_size={} train_shuffle:{} \n min_val_loss:{}'
                            .format(model,data_name,input_len,encoder_depth,decoder_depth,mask_rate,base_lr,d_model,mask_size,train_shuffle,early_stopping.val_loss_min))
                        break    

            if  model=='mae':
                break   

        if  model=='mae':
            break     
            #model_path='pre_train_model/model_ETTh2.pt'
            #torch.save(mae,model_path)