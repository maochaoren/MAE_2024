import torch 
print(torch.cuda.is_available())
import argparse
torch.autograd.set_detect_anomaly(True)
import pandas as pd
from data_loader import *
from torch.utils.data import DataLoader
from mae_model import MaskedAutoEncoder as mae_model
from MAE_BlockMask import BlockMaskedAutoEncoder as time_block_mae
from DistangledMAE import DistangleMAE as distangle_mae
from RetroMAElike_model import RetroMaskedAutoEncoder as retro_mae
from MultiDecompMAE import MultiDecompEncoder as multidec_mae
from SimMTM import SimMTM
from torchtools import EarlyStopping
#torch.backends.cudnn.enable =True
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
print(get_device())
#def get_device():
    #return 'cpu'
parser=argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='SimMTM', help='input sequence length')
parser.add_argument('--data_set', type=str, default='hour1', help='type of dataset')
parser.add_argument('--backbone', type=str, default='vanilla', help='type of backbone')
parser.add_argument('--encoder_depth', type=int, default=1, help='depth of encoder')
parser.add_argument('--input_len', type=int, default=336, help='input sequence length')
parser.add_argument('--n_head', type=int, default=8, help='number of heads in MultiHeadAttention')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
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
# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

args=parser.parse_args()
print(args)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]

dataset_dict={0:'hour1',1:'hour2',2:'ECL',3:'min1',
              4:'min2',5:'sim1',6:'sim2'}
data_set=args.data_set

model_dict={0:'mae',1:'time_block_mae',2:'retro_mae',
            3:'distangle_mae',4:'multidec_mae',5:'SimMTM'}
model=args.model

#model='multidec_mae'
#model='SimMTM'

#参数
pred_len=1
input_len_list=[args.input_len]
patience=args.patience
train_shuffle=True
epochs=args.epochs
base_lr=args.base_lr
batch_size=args.batch_size

#simMTM
is_norm=True
is_decomp=True
CI=True
mask_num=args.mask_num
tau=args.tau
mask_rate=args.mask_rate
part=args.part
mask_size=args.mask_size

backbone=args.backbone

encoder_depth=args.encoder_depth
decoder_depth=1
d_model=args.d_model
n_head=args.n_head
#d_model=d_model//4 if part=='t' else d_model

#decomp
decomp=args.decomp
topk=args.topk
window_size=args.window_size
st_sep=args.st_sep
window_list=args.window_list
#window_list=[1]
#retro_mae
#is_multidec_mae=True

#retro_mae
series_embed_len=1
enhance_decoding_list=[True]
mr_tuple_list=[[0.25,0.75]]
alpha=0.75


#mr_tuple_list=[[0.25,0.9],[0.5,0.9],[0.75,0.9],
# [0.25,0.5],[0.25,0.75],[0.5,0.5],[0.5,0.75]]

#数据集、预训练模型
#data_set='minute'
#model='retro_mae' if is_retro_mae else 'mae'
#model='mae'
#model='retro_mae'
#model='distangle_mae'

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
    for enhance_decoding in enhance_decoding_list:
        for mr_tuple in mr_tuple_list:

            mask_rate_enc=mr_tuple[0]
            mask_rate_dec=mr_tuple[1]
                
            dataset_train=Dataset(root_path=root_path,data_path=data_path,flag='train',features='M',size=[input_len,input_len,pred_len],scale=False if data_set=='sim2' else True)
            dataset_val=Dataset(root_path=root_path,data_path=data_path,flag='val',features='M',size=[input_len,input_len,pred_len],scale=False if data_set=='sim2' else True)

            data_loader_train=DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=train_shuffle,drop_last=False)
            data_loader_valid=DataLoader(dataset=dataset_val,batch_size=batch_size,shuffle=False,drop_last=False)

            if model=='mae':
                mae=mae_model(c_in=dataset_train.dim(),d_model=d_model,input_len=input_len,mask_rate=mask_rate,encoder_depth=encoder_depth,decoder_depth=decoder_depth,mask_size=mask_size,freq=freq,encoder='trans').to(get_device())
                model_save_path='pre_train_mae/'
            elif model=='time_block_mae':
                mae=time_block_mae(c_in=dataset_train.dim(),d_model=d_model,input_len=input_len,mask_rate=mask_rate,encoder_depth=encoder_depth,decoder_depth=decoder_depth,time_block=mask_size,freq=freq).to(get_device())
                model_save_path='pre_train_TBmae/'
            elif model=='retro_mae':
                mae=retro_mae(c_in=dataset_train.dim(),d_model=d_model,input_len=input_len,series_embed_len=series_embed_len,mask_rate_enc=mask_rate_enc,mask_rate_dec=mask_rate_dec,encoder_depth=encoder_depth,mask_size=1,enhance_decoding=enhance_decoding,alpha=alpha,is_norm=is_norm).to(get_device())
                model_save_path='pre_train_Retro_mae/'
            elif model=='distangle_mae':
                mae=distangle_mae(c_in=dataset_train.dim(),d_model=d_model,input_len=input_len,mask_rate_enc=mask_rate_enc,mask_rate_dec=mask_rate_dec,encoder_depth=encoder_depth,enhance_decoding=enhance_decoding,alpha=alpha).to(get_device())
                model_save_path='pre_train_distangle_mae/'
            elif model=='multidec_mae':
                mae=multidec_mae(c_in=dataset_train.dim(),d_model=d_model,n_head=n_head,input_len=input_len,window_size=window_size,st_sep=st_sep,encoder_depth=encoder_depth,mask_rate=mask_rate,is_norm=is_norm,time_block=mask_size,window_list=window_list,CI=CI,backbone=backbone,part=part).to(get_device())
                model_save_path='pre_train_multidec_mae/'
            elif model=='SimMTM':
                mae=SimMTM(c_in=dataset_train.dim(),d_model=d_model,n_head=n_head,input_len=input_len,window_size=window_size,st_sep=st_sep,topk=topk,encoder_depth=encoder_depth,mask_rate=mask_rate,is_norm=is_norm,time_block=mask_size,window_list=window_list,CI=CI,mask_num=mask_num,tau=tau,is_decomp=is_decomp,backbone=backbone
                           ,part=part).to(get_device())
                model_save_path='pre_train_SimMTM/'
            # 初始化模型、优化器和损失函数
            optimizer = torch.optim.Adam(mae.parameters(), lr=base_lr, betas=(0.9, 0.999))
            scaler = torch.cuda.amp.GradScaler()  # 使用 CUDA AMP 的 GradScaler

            # 检查是否有可用的 GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 将模型移动到设备
            mae = mae.to(device)  

            if model=='retro_mae' or model=='distangle_mae':
                path=model_save_path+data_name+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
                early_stopping=EarlyStopping(patience=patience,verbose=True,path=path)
            elif model=='SimMTM':
                path=model_save_path+data_name+'_ed'+str(encoder_depth)+'_dm'+str(d_model)+'_'+part+'_'+str(input_len)+'_ms'+str(mask_size)+'_mr'+str(mask_rate)+'_mnum'+str(mask_num)+'_'+str(is_decomp)+'_'+backbone+'.pt'
                early_stopping=EarlyStopping(patience=patience,verbose=True,path=path)
            elif model=='multidec_mae':
                path=model_save_path+data_name+'_ed'+str(encoder_depth)+'_dm'+str(d_model)+'_'+part+'_'+str(input_len)+'_ms'+str(mask_size)+'_mr'+str(mask_rate)+'_'+str(is_decomp)+'_'+backbone+'.pt'
                early_stopping=EarlyStopping(patience=patience,verbose=True,path=path)
            else:
                early_stopping=EarlyStopping(patience=patience,verbose=True,path=model_save_path+data_name+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(mask_rate)+'.pt')
            
            print(data_set)
            print(model)
            print('input_len:{}'.format(input_len))
            print('backbone:{}'.format(backbone))
            print('encoder_depth:{}'.format(encoder_depth))
            print('mask_size:{}'.format(mask_size))
            print('mask_rate:{}'.format(mask_rate))
            print('mask_num:{}'.format(mask_num))
            print('is_decomp:{}'.format(is_decomp))
            print('is_norm:{}'.format(is_norm))
            print('CI:{}'.format(CI))
            print('part:{}'.format(part))
            print('decomp:{}'.format(decomp))
            if decomp=='fft':
                print('st_sep:{}'.format(st_sep))
                print('topk:{}'.format(topk))

            else:
                print('window_list:{}'.format(window_list))

            for epoch in range(0,epochs):
                print('epoch: {}/{}'.format(epoch+1,epochs))
                #训练
                total_train_loss=0
                for data in data_loader_train:# 四项：x,y,x_mask,y_mask
                    optimizer.zero_grad()
                    mae.train()
                    x,_,x_mask,_=data
                    x=x.float()
                    x=x.to(get_device())
                    #print(x.shape)
                    #x_mask=x_mask.to(get_device())
                    with torch.cuda.amp.autocast():
                        _,loss=mae(x)
                    total_train_loss+=loss*x.shape[0]
                    #optimizer.step()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                total_train_loss/=len(data_loader_train.dataset)
                print("total_train_loss:{}".format(total_train_loss))
                #验证
                total_val_loss=0
                mae.eval() 
                for data in data_loader_valid:
                    with torch.no_grad():
                        x,_,x_mask,_=data
                        x=x.float()
                        x=x.to(get_device())
                        x_mask=x_mask.to(get_device())
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