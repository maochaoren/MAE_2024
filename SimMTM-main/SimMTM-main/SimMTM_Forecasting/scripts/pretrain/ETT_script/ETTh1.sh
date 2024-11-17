export CUDA_VISIBLE_DEVICES=4

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model SimMTM \
    --data ETTh1 \
    --features M \
    --decomp 1 \
    --seq_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 16 \
    --d_model 16 \
    --d_ff 64 \
    --positive_nums 3 \
    --mask_rate 0.5 \
    --learning_rate 0.001 \
    --batch_size 8 \
    --train_epochs 50 


