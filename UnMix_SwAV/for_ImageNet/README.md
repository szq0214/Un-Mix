# Un-Mix on SwAV for IamgeNet
This code provides a PyTorch implementation for **[Un-Mix](https://arxiv.org/abs/2003.05438)** upon **[SwAV](https://arxiv.org/abs/2006.09882)** for ImageNet dataset.


The training procedures and instructions are the same as **[SwAV code](https://github.com/facebookresearch/swav)** while simply replacing ``main_swav.py`` with ``main_swav_unmix.py``.

# Running Un-Mix with SwAV unsupervised training

## Requirements
- Python 3.6
- [PyTorch](http://pytorch.org) install = 1.4.0
- torchvision
- CUDA 10.1
- [Apex](https://github.com/NVIDIA/apex) with CUDA extension (see [how I installed apex](https://github.com/facebookresearch/swav/issues/18#issuecomment-748123838))
- Other dependencies: scipy, pandas, numpy

## Singlenode training
SwAV is very simple to implement and experiment with.
Our implementation consists in a [main_swav_unmix.py](./main_swav_unmix.py) file from which are imported the dataset definition [src/multicropdataset.py](./src/multicropdataset.py), the model architecture [src/resnet50.py](./src/resnet50.py) and some miscellaneous training utilities [src/utils.py](./src/utils.py).

For example, to train Un-Mix + SwAV baseline on a single node with 8 gpus for 400 epochs, run:
```
python -m torch.distributed.launch --nproc_per_node=8 main_swav_unmix.py \
--data_path /path/to/imagenet/train \
--epochs 400 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 15
```

## Multinode training
Distributed training is available via Slurm. We provide several [SBATCH scripts](./scripts) to train our models.
For example, to train Un-Mix + SwAV on 8 nodes and 64 GPUs with a batch size of 4096 for 800 epochs run:
```
sbatch ./scripts/unmix_swav_800ep_pretrain.sh
```
Note that you might need to remove the copyright header from the sbatch file to launch it.

**Set up `dist_url` parameter**: We refer the user to pytorch distributed documentation ([env](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization) or [file](https://pytorch.org/docs/stable/distributed.html#shared-file-system-initialization) or [tcp](https://pytorch.org/docs/stable/distributed.html#tcp-initialization)) for setting the distributed initialization method (parameter `dist_url`) correctly. In the provided sbatch files, we use the [tcp init method](https://pytorch.org/docs/stable/distributed.html#tcp-initialization) (see [\*](./scripts/unmix_swav_800ep_pretrain.sh#L17-L20) for example).

# Evaluating models

## Evaluate models: Linear classification on ImageNet
To train a supervised linear classifier on frozen features/weights on a single node with 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py \
--data_path /path/to/imagenet \
--pretrained /path/to/checkpoints/unmix_swav_800ep_pretrain.pth.tar
```
The resulting linear classifier can be downloaded [here](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar).

## Evaluate models: Semi-supervised learning on ImageNet
To reproduce our results and fine-tune a network with 1% or 10% of ImageNet labels on a single node with 8 gpus, run:
- 10% labels
```
python -m torch.distributed.launch --nproc_per_node=8 eval_semisup.py \
--data_path /path/to/imagenet \
--pretrained /path/to/checkpoints/unmix_swav_800ep_pretrain.pth.tar \
--labels_perc "10" \
--lr 0.01 \
--lr_last_layer 0.2
```
- 1% labels
```
python -m torch.distributed.launch --nproc_per_node=8 eval_semisup.py \
--data_path /path/to/imagenet \
--pretrained /path/to/checkpoints/unmix_swav_800ep_pretrain.pth.tar \
--labels_perc "1" \
--lr 0.02 \
--lr_last_layer 5
```

## Evaluate models: Transferring to Detection with DETR
[DETR](https://arxiv.org/abs/2005.12872) is a recent object detection framework that reaches competitive performance with Faster R-CNN while being conceptually simpler and trainable end-to-end. We evaluate our SwAV ResNet-50 backbone on object detection on COCO dataset using DETR framework with full fine-tuning. Here are the instructions for reproducing our experiments:

1. [Install detr](https://github.com/facebookresearch/detr#usage---object-detection) and prepare COCO dataset following [these instructions](https://github.com/facebookresearch/detr#data-preparation).

1. Apply the changes highlighted in [this gist](https://gist.github.com/mathildecaron31/bcd03b8864f7ca1aeb89dfe76a118b14#file-backbone-py-L92-L101) to [detr backbone file](https://github.com/facebookresearch/detr/blob/master/models/backbone.py) in order to load SwAV backbone instead of ImageNet supervised weights.

1. Launch training from `detr` repository with [run_with_submitit.py](https://github.com/facebookresearch/detr/blob/master/run_with_submitit.py).
```
python run_with_submitit.py --batch_size 4 --nodes 2 --lr_backbone 5e-5
```

# Common Issues

Please see [here](https://github.com/facebookresearch/swav/blob/main/README.md#common-issues) or submit a GitHub issue in this repo.

## License
See the [LICENSE](LICENSE) file for more details.


## Citation
If you find this repository useful in your research, please cite:

```
@article{caron2020unsupervised,
  title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```

```
@article{shen2020mix,
  title={Un-mix: Rethinking image mixtures for unsupervised visual representation learning},
  author={Shen, Zhiqiang and Liu, Zechun and Liu, Zhuang and Savvides, Marios and Darrell, Trevor and Xing, Eric},
  journal={arXiv preprint arXiv:2003.05438},
  year={2020}
}
```
