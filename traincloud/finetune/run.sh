# CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_endpoint 127.0.0.1:1122 --nproc_per_node 1 finetuning.py --model 'ViT-B/16' \
#     --batch_size 50 --epochs 5 --warmup_epochs 0 --accum_iter 1 \
#     --blr 9e-8 --weight_decay 0.02  --output_dir ./ckpt/pcViTB16 --dataset modelnet40 \
#     --addrotation



# CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_endpoint 127.0.0.1:1122 --nproc_per_node 1 finetuning.py --model 'ViT-B/16' \
#     --batch_size 50 --epochs 5 --warmup_epochs 2 --accum_iter 1 \
#     --blr 4e-2 --weight_decay 0.08 --output_dir ./ckpt/pcViTB16 --dataset modelnet40 


CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_endpoint 127.0.0.1:1122 --nproc_per_node 1 finetuning.py --model 'ViT-B/16' \
    --batch_size 50 --epochs 5 --warmup_epochs 2 --accum_iter 1 \
    --blr 6e-10 --weight_decay 0.08 --output_dir ./ckpt/pcViTB16 --dataset modelnet40


CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_endpoint 127.0.0.1:1122 --nproc_per_node 1 finetuning.py --model 'ViT-B/16' \
    --batch_size 50 --epochs 5 --warmup_epochs 2 --accum_iter 1 \
    --blr 7e-5 --weight_decay 0.08 --output_dir ./ckpt/pcViTB16 --dataset modelnet40 \
    --addrotation


CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_endpoint 127.0.0.1:1122 --nproc_per_node 1 finetuning.py --model 'ViT-B/16' \
    --batch_size 50 --epochs 5 --warmup_epochs 2 --accum_iter 1 \
    --blr 5e-8 --weight_decay 0.08 --output_dir ./ckpt/pcViTB16 --dataset modelnet40 \
    --addrotation

    