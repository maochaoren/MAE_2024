export CUDA_VISIBLE_DEVICES=1

patching_s=1
patch_len_s=192
patching_t=1
patch_len_t=4

#pretrain
python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id Weather \
    --model SimMTM \
    --data Weather \
    --features M \
    --decomp 1 \
    --decomp_method mov_avg \
    --st_sep  5 \
    --lpf 50 \
    --patching_s $patching_s \
    --patch_len_s $patch_len_s \
    --patching_t $patching_t \
    --patch_len_t $patch_len_t \
    --s_e_layers 2 \
    --t_e_layers 1 \
    --seq_len 384 \
    --e_layers 1 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --n_heads 8 \
    --d_model 16 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --batch_size 4 \
    --is_early_stop 1 \
    --patience 3 \
    --train_epochs 50

#finetune

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id Weather \
        --model SimMTM \
        --data Weather \
        --features M \
        --seq_len 384 \
        --label_len 48 \
        --decomp 1 \
        --decomp_method mov_avg \
        --st_sep  5 \
        --lpf 50 \
        --patching_s $patching_s \
        --patch_len_s $patch_len_s \
        --patching_t $patching_t \
        --patch_len_t $patch_len_t \
        --pred_len $pred_len \
        --e_layers 1 \
        --s_e_layers 2 \
        --t_e_layers 1 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model 16 \
        --d_ff 64 \
        --batch_size 16 \
        --learning_rate 1e-4 \
        --is_early_stop 1 \
        --patience 3
done
