export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model SimMTM \
    --data ETTh1 \
    --features M \
    --decomp 1 \
    --decomp_method mov_avg \
    --st_sep 3 \
    --lpf 50 \
    --patching_s 1 \
    --patch_len_s 48 \
    --seq_len 384 \
    --s_e_layers 2 \
    --t_e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 8 \
    --d_ff 64 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --learning_rate 7e-4 \
    --batch_size 8 \
    --is_early_stop 1 \
    --patience 5 \


bash ~/MAE_2024/SimMTM-main/SimMTM-main/SimMTM_Forecasting/scripts/finetune/ETT_script/ETTh1.sh 
 


