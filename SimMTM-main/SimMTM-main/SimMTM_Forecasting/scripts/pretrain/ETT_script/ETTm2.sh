export CUDA_VISIBLE_DEVICES=7

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model SimMTM \
    --data ETTm2 \
    --features M \
    --seq_len 336 \
    --e_layers 1 \
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


