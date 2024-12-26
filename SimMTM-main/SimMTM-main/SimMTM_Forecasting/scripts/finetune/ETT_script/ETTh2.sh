export CUDA_VISIBLE_DEVICES=2

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2 \
        --model SimMTM \
        --data ETTh2 \
        --features M \
        --decomp 1 \
        --decomp_method fft \
        --patching_t 0 \
        --patch_len_t 4 \
        --st_sep  7 \
        --lpf 50 \
        --seq_len 384 \
        --label_len 48 \
        --pred_len $pred_len \
        --s_e_layers 2 \
        --t_e_layers 1 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 4 \
        --d_model 8 \
        --d_ff 64 \
        --dropout 0.1 \
        --batch_size 32 \
        --patience 2 \
        --learning_rate 1e-4
done