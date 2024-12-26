export CUDA_VISIBLE_DEVICES=7

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2 \
        --model SimMTM \
        --data ETTm2 \
        --features M \
        --decomp 1 \
        --decomp_method fft \
        --st_sep  3 \
        --lpf 30 \
        --seq_len 384 \
        --patching_t 0 \
        --patch_len_t 48 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 16 \
        --dropout 0 \
        --batch_size 32 \
        --patience 3
done

#bash ~/MAE_2024/SimMTM-main/SimMTM-main/SimMTM_Forecasting/scripts/finetune/ETT_script/ETTm2.sh