CUDA_VISIBLE_DEVICES=1 torchrun --rdzv_endpoint 127.0.0.1:1119 --nproc_per_node 1 evaluate.py --model GPTJ6B_adapter --llama_model_path ../pretrained/gpt-j/ \
--data_path ../data/alpaca_data.json --adapter_layer 28 --adapter_len 10 --max_seq_len 384 --batch_size 2 --epochs 5 --warmup_epochs 2 \
--blr  5e-2 --weight_decay 0.07 --output_dir ./eval/gpt_dramaqa3 --dataset dramaqa --accum_iter 16 --vaq \
--resume ./checkpoint/gpt_dramaqa3/checkpoint_best.pth