import torch 
import argparse
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data_loader import *
from torch.utils.data import DataLoader
from torchtools import EarlyStopping
from fine_tuning_model_new import FineTuningModelNew
from fine_tuning_model_distangle import FineTuningModelDistangle
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

parser=argparse.ArgumentParser()

parser.add_argument('--model', type=str, default=336, help='input sequence length')
parser.add_argument('--data_set', type=str, default='hour1', help='type of dataset')
parser.add_argument('--backbone', type=str, default='vanilla', help='type of backbone')
parser.add_argument('--encoder_depth', type=int, default=1, help='depth of encoder')
parser.add_argument('--input_len', type=int, default=336, help='input sequence length')
parser.add_argument('--n_head', type=int, default=8, help='number of heads in MultiHeadAttention')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--mask_size', type=int, default=1, help='mask size')
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask rate')
parser.add_argument('--mask_num', type=int, default=3, help='mask number')
parser.add_argument('--tau', type=float, default=0.2, help='tau')
parser.add_argument('--decomp', type=str, default='fft', help='type of decomposition')
parser.add_argument('--st_sep', type=float, default=3.5, help='separation of seasonality and trend')
parser.add_argument('--topk', type=int, default=25, help='topk')
parser.add_argument('--window_size', type=int, default=24*7+1, help='window size of moving average')
parser.add_argument('--base_lr', type=float, default=1e-3, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--patience', type=int, default=3, help='patience')
parser.add_argument('--epochs', type=int, default=20, help='epochs')
parser.add_argument('--window_list', type=list, default=[24+1,12+1,1], help='window size of moving average')
parser.add_argument('--part', type=str, default='s', help='part')
parser.add_argument('--t_model', type=str, default='Linear', help='type of t_model')
parser.add_argument('--random_init', type=int, default=1, help='random init')
parser.add_argument('--frozen_num', type=int, default=0, help='frozen num')
# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

args=parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]

print(args)
dataset_dict={0:'hour1',1:'hour2',2:'ECL',3:'min1',
              4:'min2',5:'sim1',6:'sim2'}
data_set=args.data_set

#参数
input_len_list=[args.input_len]
pre_len_list=[96,192,336,720]
#pre_len_list=[720]
frozen_num=args.frozen_num
random_init=args.random_init


mask_size=args.mask_size
mask_rate=args.mask_rate
mask_num=args.mask_num

is_norm=True
use_decoder=False
CI=True
part=args.part

decomp=args.decomp
window_size=args.window_size
dom_season=96 if data_set=='min1' or data_set=='min2' else 24
st_sep=args.st_sep
topk=args.topk
lpf=0
is_decomp=True

d_model=args.d_model
encoder_depth=args.encoder_depth
n_head=args.n_head

#d_model=d_model//4 if part=='t' else d_model
model_dict={0:'mae',1:'time_block_mae',2:'retro_mae',
            3:'distangle_mae',4:'multidec_mae',5:'SimMTM'}
model=args.model


backbone_dict={0:'res',1:'vanilla'}
backbone=args.backbone

t_model_dict={0:'transformer',1:'Linear'} 
t_model=args.t_model

base_lr=args.base_lr
epochs=args.epochs
batch_size=args.batch_size
patience=args.patience
iters=1

is_plt=True
#data_set='minute'


enhance_decoding_list=[True]
#mr_tuple_list=[[0.25,0.9],[0.5,0.9],[0.75,0.9],[0.25,0.5],[0.25,0.75],[0.5,0.5],[0.5,0.75]]
mr_tuple_list=[[0.25,0.75]]


loss_dict={}

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


for input_len in input_len_list:

    for pred_len in pre_len_list:

        size_tuple=[input_len,input_len,pred_len]

        dataset_train=Dataset(root_path=root_path,data_path=data_path,flag='train',features='M',size=[input_len,input_len,pred_len],scale=False if data_set=='sim2' else True)
        dataset_val=Dataset(root_path=root_path,data_path=data_path,flag='val',features='M',size=[input_len,input_len,pred_len],scale=False if data_set=='sim2' else True)

        #dataset=Dataset(root_path='ETT',data_path=data_path,flag='train',features='M',size=size_tuple,freq=freq,scale=True)
        data_loader_ft=DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True,drop_last=False)
        data_loader_valid=DataLoader(dataset=dataset_val,batch_size=batch_size,shuffle=False,drop_last=False)

        for enhance_decoding in enhance_decoding_list:

            for mr_tuple in mr_tuple_list:
                mask_rate_enc=mr_tuple[0]
                mask_rate_dec=mr_tuple[1]

                avg_val_loss=0
                if model=='mae':
                    mae_encoder_path='pre_train_mae/'+data_name+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(mask_rate)+'.pt'
                    ft_path='fine_tuning_mae/'
                elif model=='time_block_mae':
                    mae_encoder_path='pre_train_TBmae/'+data_name+'_'+str(input_len)+'_'+str(mask_size)+'.pt'
                    ft_path='fine_tuning_BTmae/'
                elif model=='retro_mae':
                    mae_encoder_path='pre_train_Retro_mae/'+data_name+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
                    ft_path='fine_tuning_Retromae/'
                elif model=='distangle_mae':
                    mae_encoder_path='pre_train_distangle_mae/'+data_name+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
                    ft_path='fine_tuning_distangle_mae/'
                elif model=='multidec_mae':
                    mae_encoder_path='pre_train_multidec_mae/'+data_name+'_ed'+str(encoder_depth)+'_dm'+str(d_model)+'_'+part+'_'+str(input_len)+'_ms'+str(mask_size)+'_mr'+str(mask_rate)+'_'+str(is_decomp)+'_'+backbone+'.pt'
                    ft_path='fine_tuning_multidec_mae/'
                elif model=='SimMTM':
                    mae_encoder_path='pre_train_SimMTM/'+data_name+'_ed'+str(encoder_depth)+'_dm'+str(d_model)+'_'+part+'_'+str(input_len)+'_ms'+str(mask_size)+'_mr'+str(mask_rate)+'_mnum'+str(mask_num)+'_'+str(is_decomp)+'_'+backbone+'.pt'
                    ft_path='fine_tuning_SimMTM/'

                #use_cls_token=False if model!='retro_mae' else True
                use_cls_token=False

                for iter in range(0,iters):#多次训练取平均
                    mae_encoder=None
                    if not(random_init):
                        mae=torch.load(mae_encoder_path)

                        if model=='multidec_mae' or model=='SimMTM':
                            mae_encoder=mae
                        #frozen
                        for param in mae_encoder.embedding_enc_t.parameters():
                            param.requires_grad=False
                        for param in mae_encoder.embedding_enc_s.parameters():
                            param.requires_grad=False

                        count=0
                        for layers in mae_encoder.encoder_s.attn_layers:
                            if count==frozen_num:
                                break
                            for param in layers.parameters():
                                param.requires_grad=False
                            count=count+1
                        count=0
                        for layers in mae_encoder.encoder_t.attn_layers:
                            if count==frozen_num:
                                break
                            for param in layers.parameters():
                                param.requires_grad=False
                            count=count+1

                    if model=='distangle_mae':
                        ft_model=FineTuningModelDistangle(c_in=dataset_train.dim(),d_model=d_model,input_len=size_tuple[0],series_embed_len=1,distangle_mae_encoder=mae,mask_size=mask_size,pred_len=size_tuple[2],
                                    use_cls_token=use_cls_token).to(get_device())
                    elif model=='multidec_mae' or model=='SimMTM':
                        ft_model=FineTuningModelNew(c_in=dataset_train.dim(),encoder_depth=encoder_depth,d_model=d_model,dom_season=dom_season,n_head=n_head,input_len=size_tuple[0],series_embed_len=1,mae_encoder=mae_encoder,pred_len=size_tuple[2],
                                    use_cls_token=use_cls_token,is_norm=is_norm,is_decomp=is_decomp,frozen_num=frozen_num,window_size=window_size,use_decoder=use_decoder,random_init=random_init,CI=CI,backbone=backbone,part=part,t_model=t_model,decomp=decomp,st_sep=st_sep,topk=topk,lpf=lpf).to(get_device())
                    else:
                        ft_model=FineTuningModelNew(c_in=dataset_train.dim(),d_model=d_model,input_len=size_tuple[0],series_embed_len=1,mae_encoder=mae_encoder,mask_size=mask_size,pred_len=size_tuple[2],
                                    is_mae=True,use_cls_token=use_cls_token,is_norm=is_norm,is_decomp=False).to(get_device())
                    
                    optimizer=torch.optim.Adam(ft_model.parameters(),lr=base_lr,betas=(0.9,0.999))
                    scaler=torch.cuda.amp.GradScaler()

                    #early_stopping=EarlyStopping(patience=4,verbose=True,path='ft_ETTm1_784_'+str(pred_len)+'.pt') 
                    if model=='retro_mae' or model=='distangle_mae':  
                        early_stopping=EarlyStopping(patience=patience,verbose=True,path=ft_path+'ft_'+data_name+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt',delta=5e-4) 
                    elif model=='SimMTM':
                        early_stopping=EarlyStopping(patience=patience,verbose=True,path=ft_path+'ft_'+data_name+'_ed'+str(encoder_depth)+'_dm'+str(d_model)+'_'+part+'_'+str(input_len)+'_'+str(pred_len)+'_ms'+str(mask_size)+'_mr'+str(mask_rate)+'_mnum'+str(mask_num)+'_'+str(is_decomp)+'_'+backbone+'.pt',delta=5e-4)
                    elif model=='multidec_mae':
                        early_stopping=EarlyStopping(patience=patience,verbose=True,path=ft_path+'ft_'+data_name+'_ed'+str(encoder_depth)+'_dm'+str(d_model)+'_'+part+'_'+str(input_len)+'_'+str(pred_len)+'_ms'+str(mask_size)+'_mr'+str(mask_rate)+'_'+str(is_decomp)+'_'+backbone+'.pt',delta=5e-4)
                    else:
                        early_stopping=EarlyStopping(patience=patience,verbose=True,path=ft_path+'ft_'+data_name+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(mask_rate)+'.pt',delta=5e-4) 
                    
                    for epoch in range(0,epochs):
                        #print('epoch: {}/{}'.format(epoch+1,epochs))
                        #训练
                        total_train_loss=0
                        for data in data_loader_ft:# 四项：x,y,x_mask,y_mask
                            with torch.cuda.amp.autocast(enabled=False):
                                ft_model.train()
                                optimizer.zero_grad()
                                x,y,x_mask,y_mask=data
                                x=x.float()
                                y=y.float()
                                x=x.to(get_device())
                                x_mask=x_mask.to(get_device())
                                y=y.to(get_device())
                                y_mask=y_mask.to(get_device())
                                x_label=y[:,:size_tuple[1],:]
                                x_label_mask=y_mask[:,:size_tuple[1],:]
                                #print(x_label_mask.shape)
                                x_pred=y[:,size_tuple[1]:,:]
                                #print(x_pred.shape)
                                #print(y.shape)
                                x_pred_mask=y_mask[:,size_tuple[1]:,:]
                                #x=x.float()
                                if is_decomp:
                                    loss,loss_s,loss_t,pred=ft_model(x,x_pred)
                                #loss,pred=ft_model(x,x_pred)
                                    total_train_loss+=loss*x.shape[0]
                                else:
                                    loss,pred=ft_model(x,x_pred)
                                    total_train_loss+=loss*x.shape[0]
                                #print(loss)
                                #loss.backward()
                                #optimizer.step()
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()

                        total_train_loss/=len(data_loader_ft.dataset)
                        print("total_train_loss:{}".format(total_train_loss))

                        #验证
                        total_val_loss=0
                        total_val_loss_t=0
                        total_val_loss_s=0
                        ft_model.eval() 
                        for data in data_loader_valid:
                            with torch.no_grad():
                                x,y,x_mask,y_mask=data
                                x=x.float().to(get_device())
                                x_mask=x_mask.to(get_device())
                                y=y.float().to(get_device())
                                y_mask=y_mask.to(get_device())

                                x_label=y[:,:size_tuple[1],:]
                                x_label_mask=y_mask[:,:size_tuple[1],:]
                                x_pred=y[:,size_tuple[1]:,:]
                                x_pred_mask=y_mask[:,size_tuple[1]:,:]
                                if is_decomp:
                                    loss,loss_s,loss_t,pred=ft_model(x,x_pred)

                                    total_val_loss+=loss*x.shape[0]
                                    total_val_loss_s+=loss_s*x.shape[0]
                                    total_val_loss_t+=loss_t*x.shape[0]
                                else:
                                    loss,pred=ft_model(x,x_pred)
                                    total_val_loss+=loss*x.shape[0]
                                    

                        total_val_loss/=len(data_loader_valid.dataset)
                        total_val_loss_s/=len(data_loader_valid.dataset)
                        total_val_loss_t/=len(data_loader_valid.dataset)
                        
                        #print('total_val_loss__:{}'.format(total_val_loss))
                        #print('total_val_loss_s:{}'.format(total_val_loss_s))
                        #print('total_val_loss_t:{}'.format(total_val_loss_t))
                        early_stopping(total_val_loss,ft_model)
                        if early_stopping.early_stop:
                            #print('early_stopping, best val_loss:{}'.format(early_stopping.val_loss_min))
                            break  

                    #test
                    '''if model=='retro_mae' or model=='distangle_mae':
                        model_path=ft_path+'ft_'+data_name+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
                    elif model=='SimMTM':
                        model_path=ft_path+'ft_'+data_name+'_ed'+str(encoder_depth)+'_dm'+str(d_model)+'_'+part+'_'+str(input_len)+'_'+str(pred_len)+'_ms'+str(mask_size)+'_mr'+str(mask_rate)+'_mnum'+str(mask_num)+'_'+str(is_decomp)+'_'+backbone+'.pt'
                    elif model=='multidec_mae':
                        model_path=ft_path+'ft_'+data_name+'_ed'+str(encoder_depth)+'_dm'+str(d_model)+'_'+part+'_'+str(input_len)+'_'+str(pred_len)+'_ms'+str(mask_size)+'_mr'+str(mask_rate)+'_'+str(is_decomp)+'_'+backbone+'.pt'
                    else:
                        model_path=ft_path+'ft_'+data_name+'_'+str(input_len)+'_'+str(pred_len)+'_'+str(mask_rate)+'.pt'''
                    
                    ft_model=early_stopping.best_model

                    dataset=Dataset(root_path=root_path,data_path=data_path,flag='test',features='M',size=size_tuple,scale=False if data_set=='sim2' else True)
                    data_loader_valid=DataLoader(dataset=dataset,batch_size=1 if is_plt else 32,shuffle=False,drop_last=True)
                    plot_list=np.arange(0,dataset.dim()).tolist()
                    ft_model.eval()
                    total_val_loss=0
                    total_val_loss_s=0
                    total_val_loss_t=0

                    dim_val_loss=np.zeros(dataset.dim())

                    for step,(data) in enumerate(data_loader_valid):
                        with torch.no_grad():
                            x,y,x_mask,y_mask=data
                            x=x.float().to(get_device())
                            x_mask=x_mask.to(get_device())
                            y=y.float().to(get_device())
                            y_mask=y_mask.to(get_device())
                            x_label=y[:,:size_tuple[1],:]
                            x_label_mask=y_mask[:,:size_tuple[1],:]
                            x_pred=y[:,size_tuple[1]:,:]
                            x_pred_mask=y_mask[:,size_tuple[1]:,:]
                            if is_decomp:
                                criterion=nn.MSELoss()
                                loss,loss_s,loss_t,pred=ft_model(x,x_pred)
                                loss_=criterion(pred,x_pred)
                                total_val_loss+=loss_*x.shape[0]
                                total_val_loss_s+=loss_s*x.shape[0]
                                total_val_loss_t+=loss_t*x.shape[0]
                            else:
                                loss,pred=ft_model(x,x_pred)
                                total_val_loss+=loss*x.shape[0]
                            #print('eval_loss:{}'.format(loss))
                            for dim_show in range(0,dataset.dim()):
                                dim_val_loss[dim_show]+=ft_model.dim_loss(pred,x_pred,dim_show)*x.shape[0]
                        #绘图 预测红色 原数据蓝色
                        if is_plt:
                            if model=='retro_mae':
                                img_path='img/retro_mae/'+data_name+'/'
                            elif model=='distangle_mae':
                                img_path='img/distangle_mae/'+data_name+'/'
                            elif model=='multidec_mae':
                                img_path='img/distangle_mae/'+data_name+'/'
                            elif model=='SimMTM':
                                img_path='img/SimMTM/'+data_name+'/'
                            else:
                                img_path='img/mae/'+data_name+'/'

                            if step%pred_len==0:
                                y=y[:,size_tuple[1]:,:]

                                #for dim_show in range(0,dataset.dim()):
                                for dim_show in plot_list:
                                    x=np.arange(pred_len)
                                    y_pred=pred[:,:,dim_show].squeeze(0).cpu().numpy()
                                    y_label=y[:,:,dim_show].squeeze(0).cpu().numpy()
                                    dim_loss=ft_model.dim_loss(pred,x_pred,dim_show)
                                    
                                    '''if part == 't' or part == 'both':
                                        plt.plot(x,y_pred_t,color='red',linewidth=1)
                                   
                                    elif part=='s' or part=='both':
                                        plt.plot(x,y_pred_s,color='blue',linewidth=1)'''
                                    
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
                    #print('total_test_loss__:{}'.format(total_val_loss))
                    '''print(data_set)
                    print('pred_len:{}'.format(pred_len))
                    print('total_test_loss__:{}'.format(total_val_loss))
                    print('total_test_loss_s:{}'.format(total_val_loss_s))
                    print('total_test_loss_t:{}'.format(total_val_loss_t))'''

                    #for dim in range(0,dataset.dim()):
                        #print('dim{}: total_test_loss:{}'.format(dim,dim_val_loss[dim]))
                    
                    if part=='s':
                        avg_val_loss+=total_val_loss
                        print('test_loss:{}'.format(total_val_loss))
                        print('test_loss_s:{}'.format(total_val_loss_s))
                        print('test_loss_t:{}'.format(total_val_loss_t))
                    elif part=='t':
                        avg_val_loss+=total_val_loss_t
                        print('test_loss:{}'.format(total_val_loss_t))
                    else:
                        avg_val_loss+=total_val_loss
                        print('test_loss:{}'.format(total_val_loss))

                print(data_set)
                print(model)
                print('pred_len:{}'.format(pred_len))
                print('backbone:{}'.format(backbone))
                print('random_init:{}'.format(random_init))
                print('frozen_num:{}'.format(frozen_num))
                print('mask_size:{}'.format(mask_size))
                print('mask_rate:{}'.format(mask_rate))
                print('mask_num:{}'.format(mask_num))
                print('decomp:{}'.format(decomp))
                if decomp=='fft':
                    print('st_sep:{}'.format(st_sep))
                    print('topk:{}'.format(topk))
                    print('lpf:{}'.format(lpf))
                else:
                    print('window_size:{}'.format(window_size))
                print('is_decomp:{}'.format(is_decomp))
                print('is_norm:{}'.format(is_norm))
                print('CI:{}'.format(CI))
                print('d_model:{}'.format(d_model))
                print('encoder_depth:{}'.format(encoder_depth))
                print('part:{}'.format(part))
                print('t_model:{}'.format(t_model))

                avg_val_loss/=iters 
                print('avg_val_loss:{}'.format(avg_val_loss))
                loss_dict[pred_len]=avg_val_loss.cpu()

                if model == 'retro_mae' or 'distangle_mae':
                    print('model={} enhance_decoding={} mask_rate_enc={} mask_rate_dec={} data={} input_len={} pred_len={} mask_size={} lr={}\n avg_val_loss:{}'
                      .format(model,enhance_decoding,mask_rate_enc,mask_rate_dec,data_name,input_len,pred_len,mask_size,base_lr,avg_val_loss))
                else:
                    print('model={} data={} input_len={} pred_len={} mask_size={} lr={}\n avg_val_loss:{}'
                      .format(model,data_name,input_len,pred_len,mask_size,base_lr,avg_val_loss))
                if model=='mae':
                    print(1)
                    break
            if model=='mae':
                break

    for pred_len in pre_len_list:
        print('pred_len:{} avg_val_loss:{:.3f}'.format(pred_len,loss_dict[pred_len]))  
    avg=np.average(list(loss_dict.values()))
    print('avg_loss:{:.3f}'.format(avg))         
