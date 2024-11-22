export CUDA_VISIBLE_DEVICES=6

nohup python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model SimMTM \
    --data ETTh1 \
    --features M \
    --decomp 1 \
    --patching_s 1 \
    --patch_len_s 24 \
    --seq_len 336 \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 4 \
    --d_model 4 \
    --d_ff 64 \
    --positive_nums 1 \
    --mask_rate 0.5 \
    --learning_rate 0.001 \
    --batch_size 1 \
    --train_epochs 15  \
     > ./output_log/ETTh1.log 2>&1 &


