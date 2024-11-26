export CUDA_VISIBLE_DEVICES=4

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model SimMTM \
    --data ETTm2 \
    --features M \
    --decomp 1 \
    --patching_s 0 \
    --patch_len_s 48 \
    --seq_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 8 \
    --d_ff 16 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --learning_rate 0.001 \
    --batch_size 4 \
    --train_epochs 15
