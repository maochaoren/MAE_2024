export CUDA_VISIBLE_DEVICES=6
patching_s=1
patch_len_s=192
patching_t=1
patch_len_t=8


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
    --patching_s $patching_s \
    --patch_len_s $patch_len_s \
    --patching_t $patching_t \
    --patch_len_t $patch_len_t \
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
    --learning_rate 1e-3 \
    --batch_size 8 \
    --is_early_stop 1 \
    --patience 4 \


for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model SimMTM \
        --data ETTh1 \
        --features M \
        --decomp 1 \
        --decomp_method mov_avg \
        --patching_s $patching_s \
        --patch_len_s $patch_len_s \
        --patching_t $patching_t \
        --patch_len_t $patch_len_t \
        --st_sep  3 \
        --lpf 50 \
        --seq_len 384 \
        --label_len 48 \
        --pred_len $pred_len \
        --s_e_layers 2 \
        --t_e_layers 1 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 64 \
        --dropout 0 \
        --batch_size 32 \
        --is_early_stop 0 \
        --patience 3 \
        --train_epochs 6 \
        --learning_rate 1e-4  
done
#bash ~/MAE_2024/SimMTM-main/SimMTM-main/SimMTM_Forecasting/scripts/finetune/ETT_script/ETTh1.sh 
 


