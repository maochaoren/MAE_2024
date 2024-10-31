import torch 
from data_loader import Dataset_ETT_hour,Dataset_ETT_minute
from torch.utils.data import DataLoader
from torchtools import EarlyStopping
from transformer_model import TransformerModel
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'
#超参数
base_lr=1e-3
epochs=10
size=[96,48,48]

data_loader_train=DataLoader(dataset=Dataset_ETT_hour(root_path='ETT',data_path='ETTh2.csv',flag='train',features='M',size=size),batch_size=64,shuffle=True,drop_last=True)
data_loader_valid=DataLoader(dataset=Dataset_ETT_hour(root_path='ETT',data_path='ETTh2.csv',flag='val',features='M',size=size),batch_size=64,shuffle=True,drop_last=True)

model=TransformerModel(c_in=7,d_model=512,encoder_depth=3,decoder_depth=1,label_len=size[1],pred_len=size[2]).to(get_device())
optimizer=torch.optim.AdamW(model.parameters(),lr=base_lr,betas=(0.9,0.95),weight_decay=0.05)
early_stopping=EarlyStopping(patience=4,verbose=True)
for epoch in range(0,epochs):
    for data in data_loader_train:# 四项：x,y,x_mask,y_mask
        model.train()
        x,y,x_mask,y_mask=data
        x=x.to(get_device())
        x_mask=x_mask.to(get_device())
        y=y.to(get_device())
        y_mask=y_mask.to(get_device())
        x_label=y[:,:size[1],:]
        x_label_mask=y_mask[:,:size[1],:]
        #print(x_label_mask.shape)
        x_pred=y[:,size[1]:,:]
        x_pred_mask=y_mask[:,size[1]:,:]
        #x=x.float()
        loss,pred=model(x,x_label,x_pred)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #验证
    total_val_loss=0
    model.eval() 
    for data in data_loader_valid:
        with torch.no_grad():
            x,y,x_mask,y_mask=data
            x=x.to(get_device())
            x_mask=x_mask.to(get_device())
            y=y.to(get_device())
            y_mask=y_mask.to(get_device())
            x_label=y[:,:size[1],:]
            x_label_mask=y_mask[:,:size[1],:]
            x_pred=y[:,size[1]:,:]
            x_pred_mask=y_mask[:,size[1]:,:]
            loss,pred=model(x,x_label,x_pred)
            print('eval_loss:{}'.format(loss))
            total_val_loss+=loss*x.shape[0]
    total_val_loss/=len(data_loader_valid.dataset)
    print('total_val_loss:{}'.format(total_val_loss))
    early_stopping(total_val_loss,model)
    if early_stopping.early_stop:
        break