export CUDA_VISIBLE_DEVICES=0
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
            --dropout 0.2 \
            --batch_size 32 \
            --is_early_stop 1 \
            --patience 3 \
            --learning_rate 1.2e-4  
    done