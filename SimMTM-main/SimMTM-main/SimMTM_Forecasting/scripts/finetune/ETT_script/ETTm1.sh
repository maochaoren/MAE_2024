export CUDA_VISIBLE_DEVICES=1

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1 \
        --model SimMTM \
        --data ETTm1 \
        --features M \
        --decomp 1 \
        --decomp_method fft \
        --patching_t 0 \
        --patch_len_t 48 \
        --st_sep  3 \
        --lpf 50 \
        --seq_len 384 \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 64 \
        --dropout 0 \
        --batch_size 32 \
        --patience 3
done
