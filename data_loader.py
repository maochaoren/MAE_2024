import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None,stl=False):#OT:单变量预测时的预测目标 #stl:是否趋势分解
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]

        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        #print(self.data_x)

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def dim(self):
        if self.features=='S':
            return 1
        else:
            return 7


class Dataset_Sim(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='sim_data.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None,stl=False):#OT:单变量预测时的预测目标 #stl:是否趋势分解
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, int(5000*0.6) - self.seq_len, int(5000*0.8) - self.seq_len]
        border2s = [int(5000*0.6),int(5000*0.8),int(5000*1)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]

        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        #print(self.data_x)

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def dim(self):
        if self.features=='S':
            return 1
        else:
            return 3


class Dataset_ETT_hour_disentangle(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S',  
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h',mode='t'):#OT:单变量预测时的预测目标 #stl:是否趋势分解
        # size [seq_len, label_len, pred_len]
        # info
        # init
        assert flag in ['train', 'test', 'val']
        self.flag=flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.data_name=root_path.split('_')[0]
        self.root_path = root_path
        self.mode=mode# t / s
        self.data_path_t = self.data_name+'_t_full.csv'
        self.data_path_s = self.data_name+'_s_full.csv'
        self.__read_data__()

    def __read_data__(self):
        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        self.scaler = StandardScaler()
        df_t = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path_t))
        df_s = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path_s))
        
        if self.scale:
            train_data_t = df_t[border1s[0]:border2s[0]]
            train_data_s = df_s[border1s[0]:border2s[0]]
            self.scaler.fit(train_data_t.values)
            data_t = self.scaler.transform(df_t.values)
            self.scaler.fit(train_data_s.values)
            data_s = self.scaler.transform(df_s.values)
        else:
            data_t = df_t.values
            data_s = df_s.values

        #print(data_t.shape)
        self.data_t = data_t[border1:border2]
        self.data_s = data_s[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_s_x = self.data_s[s_begin:s_end]
        seq_s_y = self.data_s[r_begin:r_end]
        seq_t_x = self.data_t[s_begin:s_end]
        seq_t_y = self.data_t[r_begin:r_end]

        #print(self.data_t.shape)

        if self.mode=='t':
            return seq_t_x, seq_t_y
        elif self.mode=='s':
            return seq_s_x, seq_s_y
        elif self.mode=='both':
            return seq_s_x, seq_s_y,seq_t_x, seq_t_y
    
    def __len__(self):
        #print(len(self.data_t))
        return len(self.data_t) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def dim(self):
        if self.features=='S':
            return 1
        else:
            return 7



class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def dim(self):
        if self.features=='S':
            return 1
        else:
            return 7


class Dataset_ECL(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ECL.csv', 
                 target='MT_320', scale=True, inverse=False, timeenc=0, freq='h', cols=None):#OT:单变量预测时的预测目标
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        #train:val:test=15 3 4
        border1s = [0, 15*30*24 - self.seq_len, 15*30*24+3*30*24 - self.seq_len]
        border2s = [15*30*24, 15*30*24+3*30*24, 15*30*24+7*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns.difference(['date','MT_320'])
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def dim(self):
        if self.features=='S':
            return 1
        else:
            return 320



class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_UCR(Dataset):
    def __init__(self, data_name, flag='train', scale=True):
        # size [seq_len, label_len, pred_len]
        # info
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':'TRAIN', 'val':'VAL', 'test':'TEST'}
        self.set_type = type_map[flag]
        self.data_name=data_name
        self.scale = scale
        self.data_path = 'UCRArchive_2018/'+str(self.data_name)+'/'+data_name+'_'+str(self.set_type)+'.tsv'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path,sep='\t',header=None).to_numpy(dtype='float')
        #print(df_raw)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 

        if self.scale:
            train_data = df_raw[:,1:]
            #print(df_raw.shape)
            self.scaler.fit(train_data)
            data = self.scaler.transform(train_data)
        else:
            data = df_raw[:,1:]

        #print(data)
        self.data_x = np.expand_dims(data,axis=-1)

        if df_raw[:,:1].min()>=1:        
            label=df_raw[:,:1]-df_raw[:,:1].min() 

        elif df_raw[:,:1].min()==0:
            label=df_raw[:,:1]

        elif df_raw[:,:1].min()==-1:
            label=df_raw[:,:1]
            for i in range(0,label.shape[0]):
                label[i,0]=0 if label[i,0]==-1 else 0
                
        self.data_y=np.squeeze(label,axis=-1)

    def __getitem__(self, index):

        seq_x = self.data_x[index,:,:]
        seq_y=self.data_y[index]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) 

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def input_len(self):
        return self.data_x.shape[1]
    
    def c_in(self):
        return self.data_x.shape[2]
    
    def cls_num(self):
        return int(self.data_y.max())+1


class Dataset_HAR(Dataset):
    def __init__(self, data_name, flag='train', scale=True):
        # size [seq_len, label_len, pred_len]
        # info
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':'train', 'val':'val', 'test':'test'}
        self.set_type = type_map[flag]
        self.scale = scale
        self.data_path = 'UCI_HAR_Dataset/'+str(self.set_type)+'.pt'
        self.__read_data__()

    def __read_data__(self):
        data=torch.load(self.data_path)
        #print(df_raw)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        #print(data["samples"].shape)
        self.data_x=data["samples"].transpose(1,2)
        self.data_y=data["labels"]

    def __getitem__(self, index):

        seq_x = self.data_x[index,:,:]
        seq_y=self.data_y[index]

        return seq_x, seq_y
    
    def __len__(self):
        return self.data_x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def c_in(self):
        return self.data_x.shape[2]
    
    def input_len(self):
        return self.data_x.shape[1]
    
    def cls_num(self):
        return 6
    
class Dataset_epilepsy(Dataset):
    def __init__(self, data_name, flag='train', scale=True):
        # size [seq_len, label_len, pred_len]
        # info
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':'train', 'val':'val', 'test':'test'}
        self.set_type = type_map[flag]
        self.scale = scale
        self.data_path = 'epilepsy/'+str(self.set_type)+'.pt'
        self.__read_data__()

    def __read_data__(self):
        data=torch.load(self.data_path)
        #print(df_raw)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 

        self.data_x=data["samples"].transpose(1,2)
        self.data_y=data["labels"]

    def __getitem__(self, index):

        seq_x = self.data_x[index,:,:]
        seq_y=self.data_y[index]

        return seq_x, seq_y
    
    def __len__(self):
        return self.data_x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def input_len(self):
        return self.data_x.shape[1]
    
    def c_in(self):
        return self.data_x.shape[2]
    
    def cls_num(self):
        return 2



