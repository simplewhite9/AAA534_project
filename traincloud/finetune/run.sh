CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_endpoint 127.0.0.1:1122 --nproc_per_node 1 finetuning.py --model 'ViT-B/16' \
    --data_path ../data/alpaca_data.json --adapter_layer 32 --adapter_len 10 --max_seq_len 384 --batch_size 10 --epochs 5 --warmup_epochs 1 --accum_iter 1 \
    --blr 9e-9 --weight_decay 0.01 --output_dir ./ckpt/reason_lr7 --dataset modelnet40 