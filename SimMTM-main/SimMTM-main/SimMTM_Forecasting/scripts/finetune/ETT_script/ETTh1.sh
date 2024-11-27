for i in {0..8}
do
    export CUDA_VISIBLE_DEVICES=$i
    if python -c "import torch; assert torch.cuda.is_available() and torch.cuda.memory_reserved(0) < torch.cuda.get_device_properties(0).total_memory * 0.9" 2>/dev/null; then
        echo "Using GPU $i"
        for pred_len in 96 192 336 720; do
            python -u run.py \
                --task_name finetune \
                --is_training 1 \
                --root_path ./dataset/ETT-small/ \
                --data_path ETTh1.csv \
                --model_id ETTh1 \
                --model SimMTM \
                --data ETTh1 \
                --features M \
                --decomp 1 \
                --seq_len 336 \
                --label_len 48 \
                --pred_len $pred_len \
                --e_layers 1 \
                --enc_in 7 \
                --dec_in 7 \
                --c_out 7 \
                --n_heads 16 \
                --d_model 16 \
                --d_ff 64 \
                --learning_rate 0.0001 \
                --dropout 0.2 \
                --batch_size 4 \
                --patience 3
        done
        break
    else
        echo "GPU $i does not have enough memory, trying next GPU..."
    fi
done