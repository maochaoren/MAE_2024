import torch
import os
import numpy as np
import torch.nn.functional as F
from _eval_protocols import fit_lr,fit_svm,fit_ridge,fit_lr
from data_loader import Dataset_ETT_hour ,Dataset_ETT_minute,Dataset_UCR,Dataset_epilepsy,Dataset_HAR
from torch.utils.data import DataLoader

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def eval_max_pooling(model,x):
    x=torch.tensor(x)
    out=model.repr_gen(x.to(get_device()),pe=False)
    out=F.max_pool1d(out.transpose(1,2),kernel_size=out.size(1)).transpose(1,2)
    return out

def eval_avg_pooling(model,x):
    x=torch.tensor(x)
    out=model.repr_gen(x.to(get_device()),pe=False)
    out=F.avg_pool1d(out.transpose(1,2),kernel_size=out.size(1)).transpose(1,2)
    return out


def _eval_with_pooling(model, x, pe=True,mask=None, slicing=None, encoding_window=None,pool='avg'):
    pool_dic={'avg':F.avg_pool1d,'max':F.max_pool1d}
    pool_func=pool_dic[pool]
    
    out = model.repr_gen(x.to(get_device()),pe=pe)
    if encoding_window == 'full_series':
        if slicing is not None:
            out = out[:, slicing]
        out = pool_func(
            out.transpose(1, 2),
            kernel_size = out.size(1),
        ).transpose(1, 2)
        
    elif isinstance(encoding_window, int):
        out = pool_func(
            out.transpose(1, 2),
            kernel_size = encoding_window,
            stride = 1,
            padding = encoding_window // 2
        ).transpose(1, 2)
        if encoding_window % 2 == 0:
            out = out[:, :-1]
        if slicing is not None:
            out = out[:, slicing]
        
    elif encoding_window == 'multiscale':
        p = 0
        reprs = []
        while (1 << p) + 1 < out.size(1):
            t_out = pool_func(
                out.transpose(1, 2),
                kernel_size = (1 << (p + 1)) + 1,
                stride = 1,
                padding = 1 << p
            ).transpose(1, 2)
            if slicing is not None:
                t_out = t_out[:, slicing]
            reprs.append(t_out)
            p += 1
        out = torch.cat(reprs, dim=-1)
        
    else:
        if slicing is not None:
            out = out[:, slicing]
        
    return out.cpu()

def encode(model, dataset, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=64):
    ''' Compute representations using the model.
    
    Args:
        data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
        mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
        encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
        casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
        sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
        sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
        batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
        
    Returns:
        repr: The representations for data.
    '''
    n_samples= dataset.__len__()
    ts_l=model.input_len
    loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,drop_last=False)

    model.eval()
    
    with torch.no_grad():
        output = []
        for batch in loader:
            print('a')
            x = batch[0]
            if sliding_length is not None:
                reprs = []
                if n_samples < batch_size:
                    calc_buffer = []
                    calc_buffer_l = 0
                for i in range(0, ts_l, sliding_length):
                    l = i - sliding_padding
                    r = i + sliding_length + (sliding_padding if not casual else 0)
                    x_sliding = torch_pad_nan(
                        x[:, max(l, 0) : min(r, ts_l)],
                        left=-l if l<0 else 0,
                        right=r-ts_l if r>ts_l else 0,
                        dim=1
                    )
                    if n_samples < batch_size:
                        if calc_buffer_l + n_samples > batch_size:
                            out = _eval_with_pooling(model,
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                        calc_buffer.append(x_sliding)
                        calc_buffer_l += n_samples
                    else:
                        out = _eval_with_pooling(model,
                            x_sliding,
                            mask,
                            slicing=slice(sliding_padding, sliding_padding+sliding_length),
                            encoding_window=encoding_window
                        )
                        reprs.append(out)
                if n_samples < batch_size:
                    if calc_buffer_l > 0:
                        out = _eval_with_pooling(model,
                            torch.cat(calc_buffer, dim=0),
                            mask,
                            slicing=slice(sliding_padding, sliding_padding+sliding_length),
                            encoding_window=encoding_window
                        )
                        reprs += torch.split(out, n_samples)
                        calc_buffer = []
                        calc_buffer_l = 0
                
                out = torch.cat(reprs, dim=1)
                if encoding_window == 'full_series':
                    out = F.max_pool1d(
                        out.transpose(1, 2).contiguous(),
                        kernel_size = out.size(1),
                    ).squeeze(1)
            else:
                out = _eval_with_pooling(model,x, mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    out = out.squeeze(1)
                    
            output.append(out)
            
        output = torch.cat(output, dim=0)
        
    return output.numpy()

def eval_classification(model,dataloader_train,dataloader_test,eval_protocol='svm',use_cls_token=False,pooling='avg',pe=False):
        
    model.eval()
    pool_dic={'avg':eval_avg_pooling,'max':eval_max_pooling}

    pool_func=pool_dic[pooling]

    train_x=np.zeros((1,model.d_model))
    train_y=np.zeros((1))
    test_x=np.zeros((1,model.d_model))
    test_y=np.zeros((1))

    for data_train in dataloader_train:
        x,y=data_train
        #print(x.unsqueeze(-1).shape)
        x_=x
        with torch.no_grad():
            #print(x.shape)
            x=_eval_with_pooling(model,x.to(get_device()),encoding_window='full_series',pool=pooling,pe=pe).squeeze(1).cpu().numpy()
            if use_cls_token:
                x=model.repr_gen(x_.to(get_device()),pe=pe)[:,:1,:].squeeze(1).cpu().numpy()

        #print(x.shape)
        y=y.cpu().numpy()
        #print(y.shape)
        train_x=np.concatenate((train_x,x),axis=0)
        train_y=np.concatenate((train_y,y),axis=0)

    for data_test in dataloader_test:
        x,y=data_test
        x_=x
        with torch.no_grad():
            x=_eval_with_pooling(model,x.to(get_device()),encoding_window='full_series',pe=pe).squeeze(1).cpu().numpy()
            if use_cls_token:
                x=model.repr_gen(x_.to(get_device()),pe=pe)[:,:1,:].squeeze(1).cpu().numpy()

        y=y.cpu().numpy()
        test_x=np.concatenate((test_x,x),axis=0)
        test_y=np.concatenate((test_y,y),axis=0)

    train_x=np.delete(train_x,0,0)
    train_y=np.delete(train_y,0,0)
    test_x=np.delete(test_x,0,0)
    test_y=np.delete(test_y,0,0)


    #train_y=train_y.squeeze(1)
    #test_y=test_y.squeeze(1)

    if eval_protocol =='svm':
        fit_clf=fit_svm

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])
    #train_x=encode(model,dataset=dataset_train)
    #print(train_label.shape)
    
    clf=fit_clf(train_x,train_y)

    acc=clf.score(test_x,test_y)
    print(acc)

    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_x)
    else:
        y_score = clf.decision_function(test_x)

    return y_score


def eval_predict(model,dataset_train,dataset_test,eval_protocol='ridge',use_cls_token=False,pe=True):
        
    if eval_protocol =='ridge':
        fit_clf=fit_ridge
    elif eval_protocol =='linear':
        fit_clf=fit_lr
        
    model.eval()

    train_x=np.zeros((1,model.input_len,model.d_model))
    train_y=np.zeros((1,dataloader_train.dataset.pred_len,model.c_in))
    test_x=np.zeros((1,model.input_len,model.d_model))
    test_y=np.zeros((1,dataloader_test.dataset.pred_len,model.c_in))
                
    for data_train in dataloader_train:
        x,y,_,_=data_train
        with torch.no_grad():
            if use_cls_token==False:
                x=model.repr_gen(x.to(get_device()),pe=pe)[:,1:,:].squeeze(1).cpu().numpy().tolist()
            else:
                x=model.repr_gen(x.to(get_device()),pe=pe).squeeze(1).cpu().numpy().tolist()

        print(x)
        y=y.cpu().numpy()
        #print(y.shape)
        train_x=np.concatenate((train_x,x),axis=0)
        train_y=np.concatenate((train_y,y),axis=0)

    for data_test in dataloader_test:
        x,y,_,_=data_test
        with torch.no_grad():
            if use_cls_token==False:
                x=model.repr_gen(x.to(get_device()),pe=pe)[:,1:,:].squeeze(1).cpu().numpy()
            else:
                x=model.repr_gen(x.to(get_device()),pe=pe).squeeze(1).cpu().numpy()

        y=y.cpu().numpy()
        test_x=np.concatenate((test_x,x),axis=0)
        test_y=np.concatenate((test_y,y),axis=0)

    train_x=np.delete(train_x,0,0)
    train_y=np.delete(train_y,0,0)
    test_x=np.delete(test_x,0,0)
    test_y=np.delete(test_y,0,0)
        
    clf=fit_clf(train_x,train_y,test_x,test_y)

    acc=clf.score(test_x,test_y)
    print(acc)

    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_x)
    else:
        y_score = clf.decision_function(test_x)

    return y_score
    #train_x=encode(model,dataset=dataset_train,casual=True,sliding_length=1,sliding_padding=200)
    #print(train_x.shape)




if __name__=='__main__':
    #数据集、预训练模型
    is_retro_mae=True
    use_cls_token=True
    
    #data_set='Epilepsy'
    data_set='HAR'
    model='retro_mae' if is_retro_mae else 'mae'
    if not is_retro_mae:
        use_cls_token=False
    
    #model='mae'
    #model='time_block_mae'
    use_ft_model=True
    pe=False
    mask_size=1
    enhance_decoding_list=[True]
    mr_tuple_list=[[0.25,0.75]]
    mask_rate=0.75
    input_len_list=[200]
    pre_len_list=[12]

    if data_set=='UCR':
        Dataset=Dataset_UCR
        data_list=os.listdir('UCRArchive_2018/')
        data_name='UCR'
        data_list.sort(key=lambda x:x)
    elif data_set=='hour2':
        Dataset=Dataset_ETT_hour
        freq='h'
        data_path='ETTh2.csv'
        data_name='ETTh2'
    elif data_set=='hour1':
        Dataset=Dataset_ETT_hour
        freq='h'
        data_path='ETTh1.csv'
        data_name='ETTh1'
    elif data_set=='minute':
        Dataset=Dataset_ETT_minute
        freq='t'
        data_path='ETTm1.csv'
        data_name='ETTm1'
    elif data_set=='Epilepsy':
        Dataset=Dataset_epilepsy
        data_list=['epilepsy']
        data_name='epilepsy'
    elif data_set=='HAR':
        Dataset=Dataset_HAR
        data_list=['HAR']
        data_name='HAR'


    #分类


    #预训练、微调时加pe，svm分类时不加pe.
    #acc:0.896:微调pooling,用cls token预测。

    for data_path in data_list:
        
        #data_path='Worms'
        for enhance_decoding in enhance_decoding_list:

            for mr_tuple in mr_tuple_list:
                mask_rate_enc=mr_tuple[0]
                mask_rate_dec=mr_tuple[1]

                if is_retro_mae==True:
                    if use_ft_model==True:
                        mae_encoder_path='fine_tuning_Retro_mae_'+str(data_name)+'/ft_'+data_path+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
                    else:
                        mae_encoder_path='pre_train_Retro_mae_'+str(data_name)+'/'+data_path+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
                else:
                    if use_ft_model==True:
                        mae_encoder_path='fine_tuning_mae_'+str(data_name)+'/ft_'+data_path+'_'+str(mask_rate)+'.pt'
                    else:
                        mae_encoder_path='pre_train_mae_'+str(data_name)+'/'+data_path+'_'+str(mask_size)+'_'+str(mask_rate)+'.pt'
                    #mae_encoder_path='pre_train_Retro_mae_'+'/'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'

                dataset_train=Dataset(data_name=str(data_path),scale=True,flag='train')
                dataset_test=Dataset(data_name=str(data_path),scale=True,flag='test')

                dataloader_train=DataLoader(dataset=dataset_train,batch_size=16,shuffle=True,drop_last=False)
                dataloader_test=DataLoader(dataset=dataset_test,batch_size=16,shuffle=False,drop_last=False)

                model=torch.load(mae_encoder_path)
                model=model.to(get_device())

                #print(dataloader_train.dataset.__len__())

                score=eval_classification(model,dataloader_train,dataloader_test,eval_protocol='svm',use_cls_token=use_cls_token,pooling='avg',pe=pe)
                #print(score)
                '''if model!='retro_mae':
                    break
            if model!='retro_mae':
                break'''


        #break
    
    #ETT时间序列预测

    '''for input_len in input_len_list:


        for pred_len in pre_len_list:
            size_tuple=[input_len,input_len,pred_len]

            dataset_train=Dataset(root_path='ETT',data_path=data_path,flag='train',features='M',size=[input_len,0,pred_len],scale=True)
            dataset_val=Dataset(root_path='ETT',data_path=data_path,flag='val',features='M',size=[input_len,0,pred_len],scale=True)
            dataset_test=Dataset(root_path='ETT',data_path=data_path,flag='test',features='M',size=[input_len,0,pred_len],scale=True)

            dataloader_train=DataLoader(dataset=dataset_train,batch_size=32,shuffle=True,drop_last=False)
            dataloader_val=DataLoader(dataset=dataset_val,batch_size=32,shuffle=True,drop_last=False)
            dataloader_test=DataLoader(dataset=dataset_test,batch_size=32,shuffle=False,drop_last=False)

            for enhance_decoding in enhance_decoding_list:

                for mr_tuple in mr_tuple_list:
                    mask_rate_enc=mr_tuple[0]
                    mask_rate_dec=mr_tuple[1]

                    mae_encoder_path='pre_train_Retro_mae/'+data_name+'_'+str(input_len)+'_'+str(mask_size)+'_'+str(enhance_decoding)+'_'+str(mask_rate_enc)+'_'+str(mask_rate_dec)+'.pt'
                    model=torch.load(mae_encoder_path)

                    score=eval_predict(model,dataset_train,dataset_val,eval_protocol='ridge',use_cls_token=False,pe=True)'''
