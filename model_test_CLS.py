import torch 
import os
from data_loader import Dataset_UCR
from torch.utils.data import DataLoader
from torchtools import EarlyStopping
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'



data_set='UCR'
is_retro_mae=True
#data_set='minute'
mask_rate=0.75

enhance_decoding=False
mask_rate_enc=0.25
mask_rate_dec=0.5

model='retro_mae' if is_retro_mae else 'mae'


if data_set=='UCR':
    Dataset=Dataset_UCR
    data_list=os.listdir('UCRArchive_2018/')
    data_name='UCR'
    data_list.sort(key=lambda x:x)

if model=='mae':
    ft_path='fine_tuning_mae_UCR/'
elif model=='time_block_mae':
    ft_path='fine_tuning_BTmae_UCR/'
elif model=='retro_mae':
    ft_path='fine_tuning_Retro_mae_UCR/'


for data_path in data_list:

    data_path='CricketX'

    print(data_path)

    if model=='retro_mae':
        path=ft_path+'ft_'+data_path+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
    else:
        path=ft_path+'ft_'+data_path+'.pt'

    dataset=Dataset(data_name=str(data_path),scale=True,flag='test')
    data_loader_valid=DataLoader(dataset=dataset,batch_size=8,shuffle=False,drop_last=True)

    ft_model=torch.load(path)

    total_val_loss=0
    total_val_correct=0
    for step,(data) in enumerate(data_loader_valid):
        ft_model.eval()
        with torch.no_grad():
            x,y=data
            x=x.unsqueeze(-1).long().to(get_device())
            y=y.long().to(get_device())

            loss,pred=ft_model(x,y)
            #print("batch{}:val_loss={}".format(step,loss))
            #print('eval_loss:{}'.format(loss))
            total_val_loss+=loss*x.shape[0]
            total_val_correct+=torch.eq(pred.argmax(dim=1),y).sum().float()

    total_val_acc=total_val_correct/len(data_loader_valid.dataset)
    total_val_loss/=len(data_loader_valid.dataset)

    #print('total_test_loss:{}'.format(total_val_loss))
    print("total_test_acc:{}".format(total_val_acc))

    break

