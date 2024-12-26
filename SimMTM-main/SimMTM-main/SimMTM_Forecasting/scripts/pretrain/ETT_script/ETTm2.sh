export CUDA_VISIBLE_DEVICES=7
patching_s=1
patch_len_s=192
patching_t=1
patch_len_t=12

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model SimMTM \
    --data ETTm2 \
    --features M \
    --decomp 1 \
    --decomp_method fft \
    --st_sep  3 \
    --lpf 30 \
    --patching_s $patching_s \
    --patch_len_s $patch_len_s \
    --patching_t $patching_t \
    --patch_len_t $patch_len_t \
    --seq_len 384 \
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
    --batch_size 8 \
    --is_early_stop 1 \
    --patience 3 \

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2 \
        --model SimMTM \
        --data ETTm2 \
        --features M \
        --decomp 1 \
        --decomp_method fft \
        --st_sep  3 \
        --lpf 30 \
        --seq_len 384 \
        --patching_s $patching_s \
        --patch_len_s $patch_len_s \
        --patching_t $patching_t \
        --patch_len_t $patch_len_t \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 16 \
        --dropout 0 \
        --batch_size 32 \
        --patience 3
done
