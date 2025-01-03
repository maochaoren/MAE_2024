export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL \
    --model SimMTM \
    --data ECL \
    --features M \
    --decomp 1 \
    --patching_s 1 \
    --patch_len_s 48 \
    --seq_len 336 \
    --label_len 48 \
    --e_layers 2 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model 8 \
    --d_ff 64 \
    --n_heads 8 \
    --batch_size 4 \
    --train_epochs 50 \
    --learning_rate 0.001 \
    --is_early_stop 1 \
    --patience 4 \
    --temperature 0.02
