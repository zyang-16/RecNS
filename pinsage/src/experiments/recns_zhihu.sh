export CUDA_VISIBLE_DEVICES=0
python main.py --input ../data/zhihu/ --epochs 50 --learning_rate 0.001 --strategy RecNS --user_num 16015 --item_num 44175 --khops 3 --batch_size 512 --patience 20 --save_dir ./embeddings_recns_zhihu/  

