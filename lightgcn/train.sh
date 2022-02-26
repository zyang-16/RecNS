export CUDA_VISIBLE_DEVICES=6
python LightGCN.py --strategy RecNS --dataset zhihu --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 8192 --epoch 1000 


