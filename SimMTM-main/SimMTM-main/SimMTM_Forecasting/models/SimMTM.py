import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
from utils.losses import AutomaticWeightedLoss
from utils.tools import ContrastiveWeight, AggregationRebuild
from utils.augmentations import masked_data

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series (same pad)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean #s t
    
class fft_decomp(nn.Module):
    def __init__(self,st_sep,padding_rate=9,lpf=0):
        super(fft_decomp,self).__init__()
        self.st_sep=st_sep
        self.lpf=lpf
        self.padding_rate=padding_rate
    def forward(self,x):
        _,L,_=x.shape
        padding=nn.Parameter(torch.zeros(x.shape[0],L*self.padding_rate,x.shape[2])).to(x.device)
        x=torch.cat([x,padding],dim=1)
        x_fft=torch.fft.rfft(x,dim=1)
        x_s=x_fft.clone()
        x_t=x_fft.clone()
        x_s[:,:int(self.st_sep*(self.padding_rate+1)),:]=0
        x_t=x_fft-x_s

        if self.lpf>0:
            x_s[:,int(self.lpf*(self.padding_rate+1)):,:]=0
        

        x_s=torch.fft.irfft(x_s,dim=1)[:,:L,:]
        x_t=torch.fft.irfft(x_t,dim=1)[:,:L,:]

        return x_s,x_t

class topk_fft_decomp(nn.Module):
    def __init__(self,k):
        super(topk_fft_decomp,self).__init__()
        self.k=k
    def convert_coeff(self, x, eps=1e-6):#从复数得到相位和振幅
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))#real：实部；imag：虚部。
        phase = torch.atan2(x.imag, x.real + eps)#atan2:求向量（张量）和x轴的夹角（等同tan^-1)
        return amp, phase
    
    def forward(self,x):
        x_fft=x.clone()

        x_fft=torch.fft.rfft(x_fft,dim=1)
        amp,_=self.convert_coeff(x_fft)        
        topk_values, topk_indices = torch.topk(amp, self.k, dim=1)
        #print(topk_indices)
        mask=torch.zeros_like(x_fft).float().to(x.device)
        mask.scatter_(1, topk_indices, torch.ones_like(topk_indices).float().to(x.device))
        x_fft=x_fft*mask
        #x_fft=zero_

        x_fft=torch.fft.irfft(x_fft,dim=1)

        x_res=x-x_fft

        return x_fft,x_res
class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x

class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        dimension = 64
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        x = self.pooler(x) # [(bs * n_vars) x dimension]
        return x

class Model(nn.Module):
    """
    SimMTM
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        #self.new_seq_len = (self.seq_len//configs.patch_len_s)*configs.patch_len_s
        self.label_len = configs.label_len 
        self.output_attention = configs.output_attention
        self.configs = configs
        # Decomposition
        #decomp method
        if configs.decomp_method=='mov_avg':
            self.series_decomp=series_decomp(kernel_size=configs.window_size)
        elif configs.decomp_method=='fft':
            self.series_decomp=fft_decomp(st_sep=configs.st_sep,padding_rate=9,lpf=configs.lpf)
        self.topk_freq=topk_fft_decomp(k=configs.top_k_fft)
        self.patching_s=configs.patching_s
        self.patching_t=configs.patching_t
        if self.patching_t:
            self.patch_embedding_t = nn.Linear(configs.patch_len_t, configs.d_model) #patch tst
       
        if configs.patching_t:
            self.stride=configs.patch_len_t  
            self.patch_len_t = configs.patch_len_t
            self.patch_num_t = (self.seq_len - self.patch_len_t) // self.stride + 1
            #print(self.patch_num_t)
            #print(self.patch_embedding_t)
            #raise NotImplementedError

        if self.patching_s:
            self.patch_len_s = configs.patch_len_s
        # Embedding
        if self.configs.decomp:
            self.enc_embedding_s = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.enc_embedding_t = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
            
        self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        if not configs.decomp:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
            )
        else:
            self.encoder_s = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.s_e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
            )
            self.encoder_t = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.t_e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
            )

        # Decoder
        if self.task_name == 'pretrain':
            if not configs.decomp:
                # for reconstruction
                self.projection = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)

                # for series-wise representation
                self.pooler = Pooler_Head(configs.seq_len, configs.d_model, head_dropout=configs.head_dropout)

            else:
                self.projection_s = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)
                self.projection_t = Flatten_Head(self.patch_num_t if configs.patching_t else configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)

                self.pooler_s = Pooler_Head(configs.patch_len_s if configs.patching_s else  configs.seq_len, configs.d_model, head_dropout=configs.head_dropout)
                self.pooler_t = Pooler_Head(self.patch_num_t if configs.patching_t else configs.seq_len, configs.d_model, head_dropout=configs.head_dropout)

            self.awl = AutomaticWeightedLoss(3)
            self.contrastive = ContrastiveWeight(self.configs)
            self.aggregation = AggregationRebuild(self.configs)
            self.mse = torch.nn.MSELoss()
            
        elif self.task_name == 'finetune':
            if not configs.decomp:
                self.head = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)
            else:
                self.head_s = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)
                self.head_t = Flatten_Head(self.patch_num_t if self.patching_t else configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)

    def forecast(self, x_enc, x_mark_enc):
        # data shape
        bs, seq_len, n_vars = x_enc.shape
        # normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # channel independent
        x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.reshape(-1, seq_len, 1) # x_enc: [(bs * n_vars) x seq_len x 1]
        # embedding
        #x_mark_enc = torch.repeat_interleave(x_mark_enc, repeats=n_vars, dim=0)
        #enc_out = self.enc_embedding(enc_out, x_mark_enc)
        enc_out = self.enc_embedding(x_enc) # enc_out: [(bs * n_vars) x seq_len x d_model]
        # encoder
        enc_out, attns = self.encoder(enc_out) # enc_out: [(bs * n_vars) x seq_len x d_model]
        enc_out = torch.reshape(enc_out, (bs, n_vars, seq_len, -1)) # enc_out: [bs x n_vars x seq_len x d_model]
        # decoder
        dec_out = self.head(enc_out)  # dec_out: [bs x n_vars x pred_len]
        dec_out = dec_out.permute(0, 2, 1) # dec_out: [bs x pred_len x n_vars]
        # de-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
        
    def forecast_decomp(self, x_enc, x_mark_enc):
        # data shape
        bs, seq_len, n_vars = x_enc.shape
        # normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # channel independent
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(-1, seq_len, 1)
        # decomposition
        x_enc_s, x_enc_t = self.series_decomp(x_enc)
        # embedding
        enc_out_s = self.enc_embedding(x_enc_s)
        if self.patching_t:
            x_enc_t = x_enc_t.squeeze(-1)
            x_enc_t = x_enc_t.unfold(dimension=-1, size=self.patch_len_t, step=self.stride)
            enc_out_t = self.patch_embedding_t(x_enc_t)
        else:
            enc_out_t = self.enc_embedding(x_enc_t)
        # encoder
        enc_out_s, attns = self.encoder_s(enc_out_s)
        enc_out_t, attns = self.encoder_t(enc_out_t)
        
        enc_out_s = torch.reshape(enc_out_s, (bs, n_vars, seq_len, -1))
        if self.patching_t:
            enc_out_t = torch.reshape(enc_out_t, (bs, n_vars, self.patch_num_t, -1))
        else:
            enc_out_t = torch.reshape(enc_out_t, (bs, n_vars, seq_len, -1))
        # decoder
        dec_out_s = self.head_s(enc_out_s)
        dec_out_t = self.head_t(enc_out_t)
        dec_out_s = dec_out_s.permute(0, 2, 1)
        dec_out_t = dec_out_t.permute(0, 2, 1)
        # add & de-Normalization from Non-stationary Transformer
        dec_out = dec_out_s + dec_out_t
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
            
    def pretrain_reb_agg(self, x_enc, x_mark_enc, mask):

        # data shape
        bs, seq_len, n_vars = x_enc.shape

        # normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.reshape(-1, seq_len, 1) # x_enc: [(bs * n_vars) x seq_len x 1]

        # embedding
        #x_mark_enc = torch.repeat_interleave(x_mark_enc, repeats=n_vars, dim=0)
        #enc_out = self.enc_embedding(enc_out, x_mark_enc)
        enc_out = self.enc_embedding(x_enc) # enc_out: [(bs * n_vars) x seq_len x d_model]

        # encoder
        # point-wise
        enc_out, attns = self.encoder(enc_out) # enc_out: [(bs * n_vars) x seq_len x d_model]
        enc_out = torch.reshape(enc_out, (bs, n_vars, seq_len, -1)) # enc_out: [bs x n_vars x seq_len x d_model]

        # decoder
        dec_out = self.projection(enc_out) # dec_out: [bs x n_vars x seq_len]
        dec_out = dec_out.permute(0, 2, 1) # dec_out: [bs x seq_len x n_vars]

        # de-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        # pooler
        pooler_out = self.pooler(dec_out) # pooler_out: [bs x dimension]

        # dec_out for reconstruction / pooler_out for contrastive
        return dec_out, pooler_out

    def pretrain(self, x_enc, x_mark_enc, batch_x, mask):
        # data shape
        bs, seq_len, n_vars = x_enc.shape
        # normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        # channel independent
        x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.unsqueeze(-1) # x_enc: [bs x n_vars x seq_len x 1]
        x_enc = x_enc.reshape(-1, seq_len, 1) # x_enc: [(bs * n_vars) x seq_len x 1]
        # embedding
        enc_out = self.enc_embedding(x_enc) # enc_out: [(bs * n_vars) x seq_len x d_model]
        # encoder
        # point-wise representation
        p_enc_out, attns = self.encoder(enc_out) # p_enc_out: [(bs * n_vars) x seq_len x d_model]
        # series-wise representation
        s_enc_out = self.pooler(p_enc_out) # s_enc_out: [(bs * n_vars) x dimension]
        # series weight learning
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out) # similarity_matrix: [(bs * n_vars) x (bs * n_vars)]
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out) # agg_enc_out: [(bs * n_vars) x seq_len x d_model]
        agg_enc_out = agg_enc_out.reshape(bs, n_vars, seq_len, -1) # agg_enc_out: [bs x n_vars x seq_len x d_model]
        # decoder
        dec_out = self.projection(agg_enc_out)  # dec_out: [bs x n_vars x seq_len]
        dec_out = dec_out.permute(0, 2, 1) # dec_out: [bs x seq_len x n_vars]
        # de-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        pred_batch_x = dec_out[:batch_x.shape[0]]
        # series reconstruction
        loss_rb = self.mse(pred_batch_x, batch_x.detach())
        # loss
        loss = self.awl(loss_cl, loss_rb)
        #print('loss_cl: {}, loss_rb: {}, loss: {}'.format(loss_cl.item(), loss_rb.item(), loss.item()))
        return loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x
        
        
    def pretrain_decomp(self,batch_x, x_mark_enc):

        # data shape
        bs, seq_len, n_vars = batch_x.shape
        bs *= (1+self.configs.positive_nums)
        x_enc = batch_x
        # normalization
        means = torch.mean(x_enc, 1,keepdim=True).detach()
        #means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        stdev  = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc =x_enc / stdev
        x_enc = x_enc.squeeze(1)

        #decomposition
        #print(x_enc.shape)
        x_enc_s, x_enc_t = self.series_decomp(x_enc)

        #data augmentation
        x_enc_s_m, x_s_mark_enc, mask = masked_data(x_enc_s, x_mark_enc, self.configs.mask_rate, self.configs.lm, self.configs.positive_nums,self.configs.masked_rule)
        x_enc_s = torch.cat([x_enc_s, x_enc_s_m], dim=0)
        x_enc_t_m, x_t_mark_enc, mask = masked_data(x_enc_t, x_mark_enc, self.configs.mask_rate, self.configs.lm, self.configs.positive_nums,self.configs.masked_rule,mask)
        x_enc_t = torch.cat([x_enc_t, x_enc_t_m], dim=0)

        #to device
        x_enc_s = x_enc_s.to(x_enc.device)
        x_enc_t = x_enc_t.to(x_enc.device)
        #x_s_mark_enc = x_s_mark_enc.to(x_enc.device)
        #x_t_mark_enc = x_t_mark_enc.to(x_enc.device)

        #channel independent
        x_enc_s = x_enc_s.permute(0, 2, 1)
        x_enc_t = x_enc_t.permute(0, 2, 1)
        x_enc_s = x_enc_s.unsqueeze(-1)
        x_enc_t = x_enc_t.unsqueeze(-1)
        x_enc_s = x_enc_s.reshape(-1, seq_len, 1)
        x_enc_t = x_enc_t.reshape(-1, seq_len, 1)
        #print(x_enc_s[0,:,0])
        # embedding
        enc_out_s = self.enc_embedding(x_enc_s)
        #print(enc_out_s[0,0,0])
        if self.patching_t:
            x_enc_t = x_enc_t.squeeze(-1)
            x_enc_t = x_enc_t.unfold(dimension=-1, size=self.patch_len_t, step=self.stride) #bs*n_vars x patch_num x patch_len
            enc_out_t = self.patch_embedding_t(x_enc_t) #bs*n_vars x patch_num x d_model
        else:
            enc_out_t = self.enc_embedding(x_enc_t)

        # encoder
        # point-wise representation
        p_enc_out_s, attns = self.encoder_s(enc_out_s)
        p_enc_out_t, attns = self.encoder_t(enc_out_t)
        
        # series-wise representation
        if self.patching_s:
            # p_enc_out_s : [(bs*n_vars)*patch_num x self.patch_len_s x d_model]
            #new_seq_len = (p_enc_out_s.shape[1] // self.patch_len_s) * self.patch_len_s
            #p_enc_out_s = p_enc_out_s[:, :new_seq_len, :]
            p_enc_out_s = p_enc_out_s.reshape(-1, self.patch_len_s, self.configs.d_model)
        
        s_enc_out_s = self.pooler_s(p_enc_out_s)
        s_enc_out_t = self.pooler_t(p_enc_out_t)

        # series weight learning
        loss_cl_s, similarity_matrix_s, logits_s, positives_mask_s = self.contrastive(s_enc_out_s)
        loss_cl_t, similarity_matrix_t, logits_t, positives_mask_t = self.contrastive(s_enc_out_t)

        rebuild_weight_matrix_s, agg_enc_out_s = self.aggregation(similarity_matrix_s, p_enc_out_s)
        rebuild_weight_matrix_t, agg_enc_out_t = self.aggregation(similarity_matrix_t, p_enc_out_t)

        #print(agg_enc_out_t.shape)
        agg_enc_out_s = agg_enc_out_s.reshape(bs, n_vars, seq_len, -1)
        if self.patching_t:
            agg_enc_out_t = agg_enc_out_t.reshape(bs, n_vars, self.patch_num_t, -1)
        else:
            agg_enc_out_t = agg_enc_out_t.reshape(bs, n_vars, seq_len, -1)
        #print(agg_enc_out_s.shape)
        # decoder
        dec_out_s = self.projection_s(agg_enc_out_s)
        dec_out_t = self.projection_t(agg_enc_out_t)
        dec_out_s = dec_out_s.permute(0, 2, 1)
        dec_out_t = dec_out_t.permute(0, 2, 1)

        dec_out = dec_out_s + dec_out_t

        pred_batch_x = dec_out[:batch_x.shape[0]]
        #print(pred_batch_x.shape,stdev.shape)
        # de-Normalization
        pred_batch_x = pred_batch_x * (stdev.repeat(1, self.seq_len, 1))
        pred_batch_x = pred_batch_x + (means.repeat(1, self.seq_len, 1))
        # series reconstruction
        loss_rb = self.mse(pred_batch_x, batch_x.detach())
        loss_cl = loss_cl_s+loss_cl_t

        # loss
        loss = self.awl(loss_cl, loss_rb)
        #print('loss_cl_s: {}, loss_cl_t: {}, loss_rb: {}, loss: {}'.format(loss_cl_s.item(), loss_cl_t.item(), loss_rb.item(), loss.item()))

        return loss, loss_cl, loss_rb, positives_mask_s, logits_s, rebuild_weight_matrix_s, pred_batch_x
        

    def forward(self, batch_x=None, x_mark_enc=None, mask=None):

        if self.task_name == 'pretrain':
            if not self.configs.decomp:
                return self.pretrain(batch_x, x_mark_enc)
            else:
                return self.pretrain_decomp(batch_x, x_mark_enc)
        if self.task_name == 'finetune':
            if not self.configs.decomp:
                dec_out = self.forecast(batch_x, x_mark_enc)
            else:
                dec_out = self.forecast_decomp(batch_x, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None
