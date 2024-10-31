import torch 
from data_loader import Dataset_ETT_hour_disentangle
from torch.utils.data import DataLoader
from torchtools import EarlyStopping
from fine_tuning_model_new import FineTuningModelNew
from fine_tuning_model_distangle import FineTuningModelDistangle
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

#不要使用transformer作为 fine_tune的decoder，只使用Linear即可。

data_set='hour1'

#参数
is_retro_mae=True
mode_list=['s']
is_norm=False #s norm=False
input_len_list=[96]
pre_len_list=[192,720]

frozen=True
frozen_num=3

train_shuffle=True
mask_size=1
d_model=64

base_lr=1e-4
epochs=100
patience=3
iters=1
series_embed_len=1
#data_set='minute'

model='retro_mae' if is_retro_mae else 'mae'
#model='retro_mae' if is_distangle_mae else 'mae'
#model='mae'
#model='retro_mae'
#model='distangle_mae'
#mode_list=['s','t']

enhance_decoding_list=[True]
#mr_tuple_list=[[0.25,0.9],[0.5,0.9],[0.75,0.9],[0.25,0.5],[0.25,0.75],[0.5,0.5],[0.5,0.75]]
mr_tuple_list=[[0.5,0.75]]

mask_rate=0.75

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

for input_len in input_len_list:

    for pred_len in pre_len_list:

        size_tuple=[input_len,input_len,pred_len]

        for enhance_decoding in enhance_decoding_list:

            for mr_tuple in mr_tuple_list:

                mask_rate_enc=mr_tuple[0]
                mask_rate_dec=mr_tuple[1]

                for mode in mode_list:

                    dataset_train=Dataset(root_path=root_path,flag='train',features='M',size=[input_len,input_len,pred_len],scale=False,mode=mode)
                    dataset_val=Dataset(root_path=root_path,flag='val',features='M',size=[input_len,input_len,pred_len],scale=False,mode=mode)

                    #dataset=Dataset(root_path='ETT',data_path=data_path,flag='train',features='M',size=size_tuple,freq=freq,scale=True)
                    data_loader_ft=DataLoader(dataset=dataset_train,batch_size=32,shuffle=train_shuffle,drop_last=False)
                    data_loader_valid=DataLoader(dataset=dataset_val,batch_size=32,shuffle=False,drop_last=False)
                    avg_val_loss=0
                    if model=='mae':
                        mae_encoder_path='pre_train_mae_st/'+data_name+'_'+mode+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(mask_rate)+'.pt'
                        ft_path='fine_tuning_mae_st/'
                    elif model=='time_block_mae':
                        mae_encoder_path='pre_train_TBmae/'+data_name+'_'+mode+'_'+str(input_len)+'_'+str(mask_size)+'.pt'
                        ft_path='fine_tuning_BTmae/'
                    elif model=='retro_mae':
                        mae_encoder_path='pre_train_Retro_mae_st/'+data_name+'_'+mode+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
                        ft_path='fine_tuning_retro_mae_st/'

                    mae=torch.load(mae_encoder_path)

                    if model=='retro_mae':
                        mae_encoder=mae.encoder
                    else:
                        mae_encoder=mae

                    use_cls_token=False if model!='retro_mae' else True
                    
                    if model =='distangle_mae':
                        if frozen:
                            for param in mae.conv_embed.parameters():
                                param.requires_grad=False
                            for param in mae.distangle.parameters():
                                param.requires_grad=False
                            '''count=0
                            for blk in mae.Transformer_Encoder.layers:
                                if count==frozen_num:
                                    break
                                for param in blk.parameters():
                                    param.requires_grad=False
                                count+=1'''
                            for param in mae.season_mae.parameters():
                                param.requires_grad=False
                            for param in mae.trend_mae.parameters():
                                param.requires_grad=False
                    else:
                        if frozen:
                            for param in mae_encoder.ScalarProjection_enc.parameters():
                                param.requires_grad=False

                            count=0
                            for blk in mae_encoder.Transformer_Encoder.layers:
                                if count==frozen_num:
                                    break
                                for param in blk.parameters():
                                    param.requires_grad=False
                                count+=1

                    for iter in range(0,iters):#多次训练取平均
                        if model=='retro_mae':
                            ft_model=FineTuningModelNew(c_in=dataset_train.dim(),d_model=d_model,input_len=size_tuple[0],series_embed_len=series_embed_len,mae_encoder=mae_encoder,mask_size=mask_size,pred_len=size_tuple[2],
                                        is_mae=True,use_cls_token=use_cls_token,is_norm=is_norm).to(get_device())
                        else:
                            ft_model=FineTuningModelNew(c_in=dataset_train.dim(),d_model=d_model,input_len=size_tuple[0],series_embed_len=series_embed_len,mae_encoder=mae_encoder,mask_size=mask_size,pred_len=size_tuple[2],
                                        is_mae=True,use_cls_token=False).to(get_device())
                            
                        optimizer=torch.optim.Adam(ft_model.parameters(),lr=base_lr,betas=(0.9,0.999))
                        #early_stopping=EarlyStopping(patience=4,verbose=True,path='ft_ETTm1_784_'+str(pred_len)+'.pt') 
                        if model=='retro_mae' :  
                            early_stopping=EarlyStopping(patience=patience,verbose=True,path=ft_path+'ft_'+data_name+'_'+mode+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt',delta=5e-4) 
                        else:
                            early_stopping=EarlyStopping(patience=patience,verbose=True,path=ft_path+'ft_'+data_name+'_'+mode+'_'+str(input_len)+'_'+str(pred_len)+'.pt',delta=5e-4) 
                        for epoch in range(0,epochs):
                            #print('epoch: {}/{}'.format(epoch+1,epochs))
                            #训练
                            total_train_loss=0
                            for data in data_loader_ft:# 四项：x,y,x_mask,y_mask
                                ft_model.train()
                                x,y=data
                                x=x.to(get_device())
                                y=y.to(get_device())
                                x=x.float()
                                y=y.float()
                                #print(x_label_mask.shape)
                                x_pred=y[:,size_tuple[1]:,:]
                                #print(x_pred.shape)
                                loss,pred=ft_model(x,x_pred)
                                total_train_loss+=loss*x.shape[0]
                                #print(loss)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                            total_train_loss/=len(data_loader_ft.dataset)
                            #print("total_train_loss:{}".format(total_train_loss))

                            #验证
                            total_val_loss=0
                            ft_model.eval() 
                            for data in data_loader_valid:
                                with torch.no_grad():
                                    x,y=data
                                    x=x.to(get_device())
                                    y=y.to(get_device())
                                    x=x.float()
                                    y=y.float()
                                    x_pred=y[:,size_tuple[1]:,:]
                                    loss,pred=ft_model(x,x_pred)

                                    #print('eval_loss:{}'.format(loss))
                                    '''sigma_loss=0
                                    for dim in range(0,dataset_val.dim()):
                                        sigma_loss+=ft_model.dim_loss(pred,x_pred,dim)
                                    sigma_loss/=dataset_val.dim()'''
                                    #print('sigma_loss:{}'.format(sigma_loss))

                                    total_val_loss+=loss*x.shape[0]
                            total_val_loss/=len(data_loader_valid.dataset)
                            #print('total_val_loss:{}'.format(total_val_loss))
                            early_stopping(total_val_loss,ft_model)
                            if early_stopping.early_stop:
                                #print('early_stopping, best val_loss:{}'.format(early_stopping.val_loss_min))
                                avg_val_loss+=early_stopping.val_loss_min

                                break  
                    avg_val_loss/=iters  
                    if model == 'retro_mae' or 'distangle_mae':
                        print('model={} enhance_decoding={} mask_rate_enc={} mask_rate_dec={} data={} input_len={} pred_len={} mask_size={} lr={}\n avg_val_loss:{}'
                          .format(model,enhance_decoding,mask_rate_enc,mask_rate_dec,data_name,input_len,pred_len,mask_size,base_lr,avg_val_loss))
                    else:
                        print('model={} data={} input_len={} pred_len={} mask_size={} lr={}\n avg_val_loss:{}'
                          .format(model,data_name,input_len,pred_len,mask_size,base_lr,avg_val_loss))
                if model=='mae':
                    break
            if model=='mae':
                break
                
