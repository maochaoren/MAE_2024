export CUDA_VISIBLE_DEVICES=7

python -u pre_training.py \
    --model SimMTM \
    --data_set hour1 \
    --backbone vanilla \
    --part st \
    --encoder_depth 2 \
    --input_len 336 \
    --n_head 16 \
    --d_model 16 \
    --mask_size 336 \
    --mask_rate 0.5 \
    --mask_num 3 \
    --tau 0.2 \
    --decomp fft \
    --st_sep 3.5 \
    --topk 50 \
    --window_size 169 \
    --base_lr 0.001 \
    --batch_size 4 \
    --patience 3 \
    --epochs 20 \

python -u fine_tuning.py\
    --model SimMTM \
    --data_set hour1 \
    --backbone vanilla \
    --t_model linear \
    --part st \
    --encoder_depth 2 \
    --input_len 336 \
    --n_head 16 \
    --d_model 16 \
    --mask_size 336 \
    --mask_rate 0.5 \
    --mask_num 3 \
    --tau 0.2 \
    --decomp fft \
    --st_sep 3.5 \
    --topk 50 \
    --window_size 169 \
    --base_lr 0.001 \
    --batch_size 4 \
    --patience 3 \
    --epochs 20 \
    --random_init False \
    --frozen_num 0 \
