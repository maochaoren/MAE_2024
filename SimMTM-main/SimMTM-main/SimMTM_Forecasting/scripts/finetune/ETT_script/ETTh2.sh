export CUDA_VISIBLE_DEVICES=3

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
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 1 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 16 \
        --d_model 16 \
        --d_ff 32 \
        --dropout 0.2 \
        --batch_size 4 \
        --patience 3 
done