
## Un-Mix with MoCo: Transferring to Detection

The `train_net.py` script reproduces the object detection experiments on Pascal VOC and COCO.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Convert a pre-trained MoCo model to detectron2's format:
   ```
   python3 convert-pretrain-to-detectron2.py input.pth.tar output.pkl
   ```

1. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Run training:
   ```
   python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```

### Results

Please refer to our [paper](https://arxiv.org/abs/2003.05438).
