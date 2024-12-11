export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model SimMTM \
    --data Weather \
    --features M \
    --decomp 1 \
    --decomp_method fft \
    --st_sep  5 \
    --lpf 50 \
    --patching_s 1 \
    --patch_len_s 48 \
    --seq_len 336 \
    --e_layers 1 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --n_heads 8 \
    --d_model 8 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --batch_size 8 \
    --is_early_stop 1 \
    --patience 3 \
    --train_epochs 50


