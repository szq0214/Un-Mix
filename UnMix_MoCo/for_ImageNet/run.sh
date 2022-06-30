# training
python main_moco_unmix.py --lr 0.03 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 /home/datasets/imagenet --unmix_prob 0.5 --mlp --moco-t 0.20 --moco-tm 0.20 --aug-plus --cos -j 40

# linear evaluation
# python main_lincls_unmix.py  -a resnet50  --lr 30.0  --batch-size 256   --pretrained /path/to/checkpoint_0199.pth.tar   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 /home/datasets/imagenet
