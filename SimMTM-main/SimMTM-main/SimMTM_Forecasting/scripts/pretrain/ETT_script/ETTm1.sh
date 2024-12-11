export CUDA_VISIBLE_DEVICES=7
python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1 \
    --model SimMTM \
    --data ETTm1 \
    --decomp 1 \
    --decomp_method fft \
    --st_sep  3 \
    --lpf 50 \
    --features M \
    --seq_len 384 \
    --e_layers 1 \
    --s_e_layers 2 \
    --t_e_layers 1 \
    --patching_s 1 \
    --patch_len_s 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 8 \
    --d_ff 64 \
    --positive_nums 3 \
    --mask_rate 0.5 \
    --learning_rate 0.001 \
    --batch_size 8 \
    --is_early_stop 1 \
    --patience 4 \

bash ~/MAE_2024/SimMTM-main/SimMTM-main/SimMTM_Forecasting/scripts/finetune/ETT_script/ETTm1.sh


