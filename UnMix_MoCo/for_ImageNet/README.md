## Un-Mix with MoCo



This is a PyTorch implementation of [Un-Mix](https://arxiv.org/abs/2003.05438) on [MoCo](https://arxiv.org/abs/1911.05722) for ImageNet dataset:
```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```
It also includes the implementation of the [MoCo v2 paper](https://arxiv.org/abs/2003.04297):
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```
[Un-Mix paper](https://arxiv.org/abs/2003.05438):
```
@article{shen2020mix,
  title={Un-mix: Rethinking image mixtures for unsupervised visual representation learning},
  author={Shen, Zhiqiang and Liu, Zechun and Liu, Zhuang and Savvides, Marios and Darrell, Trevor and Xing, Eric},
  journal={arXiv preprint arXiv:2003.05438},
  year={2020}
}
```

### Preparation (Following MoCo)

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo aims to be minimal modifications on that code. Check the modifications by:
```
diff main_moco_unmix.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
diff main_lincls_unmix.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco_unmix.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus.


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls_unmix.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

### Transferring to Object Detection

See [./detection](detection).


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

