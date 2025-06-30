# #!/bin/bash

### CIFAR100
# c100r34r18
nohup python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 0 --warmup 20 --epochs 120 \
--dataset cifar100 --method cpnet --lr_g 4e-3 --teacher resnet34 --student resnet18 --save_dir run/c100r34r18-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.0 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c100r34r18-cpnet-ep120 > logs/final/c100r34r18-cpnet-ep120.log 2>&1 &

# c100w402w161
nohup python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 1 --warmup 20 --epochs 120 \
--dataset cifar100 --method cpnet --lr_g 4e-3 --teacher wrn40_2 --student wrn16_1 --save_dir run/c100w402w161-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.0 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c100w402w161-cpnet-ep120 > logs/final/c100w402w161-cpnet-ep120.log 2>&1 &

# c100w402w162
nohup python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 2 --warmup 20 --epochs 120 \
--dataset cifar100 --method cpnet --lr_g 4e-3 --teacher wrn40_2 --student wrn16_2 --save_dir run/c100w402w162-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.0 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c100w402w162-cpnet-ep120 > logs/final/c100w402w162-cpnet-ep120.log 2>&1 &

# c100w402w401
nohup python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 3 --warmup 20 --epochs 120 \
--dataset cifar100 --method cpnet --lr_g 4e-3 --teacher wrn40_2 --student wrn40_1 --save_dir run/c100w402w401-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.7 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c100w402w401-cpnet-ep120 > logs/final/c100w402w401-cpnet-ep120.log 2>&1 &

# c100vgg11r18
nohup python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 4 --warmup 20 --epochs 120 \
--dataset cifar100 --method cpnet --lr_g 4e-3 --teacher vgg11 --student resnet18 --save_dir run/c100vgg11r18-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.8 --g_steps 40 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c100vgg11r18-cpnet-ep120 > logs/final/c100vgg11r18-cpnet-ep120.log 2>&1 &


### CIFAR10
# c10r34r18
nohup python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 2 --warmup 20 --epochs 120 \
--dataset cifar10 --method cpnet --lr_g 4e-3 --teacher resnet34 --student resnet18 --save_dir run/c10r34r18-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.0 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c10r34r18-cpnet-ep120 > logs/final/c10r34r18-cpnet-ep120.log 2>&1 &

c10w402w161
nohup python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 3 --warmup 20 --epochs 120 \
--dataset cifar10 --method cpnet --lr_g 4e-3 --teacher wrn40_2 --student wrn16_1 --save_dir run/c10w402w161-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.0 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c10w402w161-cpnet-ep120 > logs/final/c10w402w161-cpnet-ep120.log 2>&1 &

# c10w402w162
nohup python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 3 --warmup 20 --epochs 120 \
--dataset cifar10 --method cpnet --lr_g 4e-3 --teacher wrn40_2 --student wrn16_2 --save_dir run/c10w402w162-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.0 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c10w402w162-cpnet-ep120 > logs/final/c10w402w162-cpnet-ep120.log 2>&1 &

# c10w402w401
nohup python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 4 --warmup 20 --epochs 120 \
--dataset cifar10 --method cpnet --lr_g 4e-3 --teacher wrn40_2 --student wrn40_1 --save_dir run/c10w402w401-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.7 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c10w402w401-cpnet-ep120 > logs/final/c10w402w401-cpnet-ep120.log 2>&1 &

# c10vgg11r18
nohup python datafree_kd.py --batch_size 512 --synthesis_batch_size 400 --lr 0.2 --gpu 4 --warmup 20 --epochs 120 \
--dataset cifar10 --method cpnet --lr_g 4e-3 --teacher vgg11 --student resnet18 --save_dir run/c10vgg11r18-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.8 --g_steps 30 --g_life 10 --g_loops 2 --gwp_loops 10 \
--log_tag c10vgg11r18-cpnet-ep120 > logs/final/c10vgg11r18-cpnet-ep120.log 2>&1 &

### TinyImageNet
# tir34r18
nohup python datafree_kd.py --batch_size 256 --synthesis_batch_size 200 --lr 0.2 --gpu 7 --warmup 20 --epochs 120 \
--dataset tiny_imagenet --method cpnet --lr_g 4e-3 --teacher resnet34 --student resnet18 --save_dir run/tir34r18-cpnet \
--adv 1.33 --bn 10.0 --oh 0.5 --dist 0.0 --g_steps 60 --g_life 10 --g_loops 4 --gwp_loops 20 --kd_steps 1000 \
--log_tag tir34r18-cpnet-ep120 > logs/final/tir34r18-cpnet-ep120.log 2>&1 &
