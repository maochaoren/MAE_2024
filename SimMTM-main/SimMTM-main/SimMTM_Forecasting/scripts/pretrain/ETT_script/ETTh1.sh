export CUDA_VISIBLE_DEVICES=5

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model SimMTM \
    --data ETTh1 \
    --features M \
    --decomp 1 \
    --patching_s 0 \
    --patch_len_s 48 \
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
    --batch_size 4 \
    --patience 3

 


