
## Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation Learning

## Update (06/30/2022)

**New:** We add more implementation code and models in this repo:

- **Un-Mix** w/ [SwAV](https://arxiv.org/abs/2006.09882) on CIFAR-10 and CIFAR-100, please check our results and code out at [./UnMix_SwAV/for_CIFAR](UnMix_SwAV/for_CIFAR) (**93.0%** accuracy on CIFAR-10 and **69.9%** accuracy on CIFAR-100). 

- **Un-Mix** w/ [SwAV](https://arxiv.org/abs/2006.09882) on ImageNet-1K, please check it out at [./UnMix_SwAV/for_ImageNet](UnMix_SwAV/for_ImageNet).

 (Insights of Un-Mix on clustering-based methods like SwAV can be found [here](https://github.com/szq0214/Un-Mix/issues/7#issuecomment-1157572883).)

- **Un-Mix** w/ MoCo [V1](https://arxiv.org/abs/1911.05722), [V2](https://arxiv.org/abs/2003.04297) on ImageNet-1K, please check it out at [./UnMix_MoCo/for_ImageNet](UnMix_MoCo/for_ImageNet). 

(**Un-Mix** w/ [MoCo](https://arxiv.org/abs/1911.05722) on CIFAR-10 and CIFAR-100 has already been included in this repo.)

**Notes:** As Un-Mix implementation on CIFAR with SwAV was done in 2020, while the SwAV authors simplified their cluster assignment implementation in [April, 2021](https://github.com/facebookresearch/swav/commit/9a2dc8073884c11de691ffe734bd624a84ccd96d). Thus, our CIFAR code follows their old implementation before simplification, but the ImageNet code is based on their updated version. Nevertheless, the performance of them will not be affected. 

## Update (01/07/2022)

- **<font size=4>Un-Mix has been accepted in AAAI 2022!!</font>** Please check out our camera-ready paper on [arXiv](https://arxiv.org/pdf/2003.05438.pdf).

- We provide a demo of Un-Mix on CIFAR-10 and 100 adapted from [Colab notebook](https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb) of MoCo.

	To train an Un-Mix model with MoCo and symmetric loss, run `unmix_c10.py` or `unmix_c100.py`:
	
	
	```
	# CIFAR-10
	CUDA_VISIBLE_DEVICES=0 python unmix_c10.py 
	
	# CIFAR-100
	CUDA_VISIBLE_DEVICES=1 python unmix_c100.py 
	```

	**Results on CIFAR-10:**
	
	| Model    | epochs  | acc. (Top-1)  | weights (last) |logs| args|
	|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
	| `MoCo w/ asymmetric` | 1000 | 89.11% | [link](https://drive.google.com/file/d/1SFug83e6J2U4WsnapFxAs4UinGMcfrRZ/view?usp=sharing) |  [link](https://drive.google.com/file/d/1LX0sZmUH6Datwq3H5mPnkOL_kxDr0oWs/view?usp=sharing) |  [link](https://drive.google.com/file/d/1-CpyJh41J5cd9aKyZIQT5x9V5Oph2Lrr/view?usp=sharing) | 
	| **`+Un-Mix w/ asymmetric`** | 1000 | **91.34%**  | [link](https://drive.google.com/file/d/1kuRZIcyWGHYYge1Y_CxkBIuaUxVmdLTD/view?usp=sharing) |  [link](https://drive.google.com/file/d/1s8wvuFdRiw6ZreRotyvHFXyNWpCee5ka/view?usp=sharing) |  [link](https://drive.google.com/file/d/1d2cOGhKhRWUYC19HCO-ZopjvPMof6wGO/view?usp=sharing) | 
	|   |   |  |  |  |
	| `MoCo w/ symmetric` |  1000 | 90.49% | [link](https://drive.google.com/file/d/1VribpdEWZ-MoyKRw2YBV9FY-AvHY3lsy/view?usp=sharing) |  [link](https://drive.google.com/file/d/11ptyy0XC7zthsNVY_xJKxIfNQ-Ikn55e/view?usp=sharing) | [link](https://drive.google.com/file/d/1I6eRUJT0AG_mFDodgimQfomBy4VAXeCe/view?usp=sharing) | 
	| **`+Un-Mix w/ symmetric`** | 1000 |**92.25%**  | [link](https://drive.google.com/file/d/1hUq3m7c6a6Pg1faLqdNLYPDeQ4Py2ZJz/view?usp=sharing) |  [link](https://drive.google.com/file/d/1vrJaKb9QzEt0P0aYe59LsMAOtn8XLABA/view?usp=sharing) | [link](https://drive.google.com/file/d/1yy2SmlJiRubyYwLbyiv7DE9Y7e8EAKk-/view?usp=sharing) | 
	
	**Results on CIFAR-100:**
	
	| Model    | epochs  | acc. (Top-1)  | weights (last) |logs|args|
	|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
	| `MoCo w/ asymmetric` | 1000 | 63.64% | [link](https://drive.google.com/file/d/1KWjk1366Z9oPqNlCtvjW2csOHe4KIuqX/view?usp=sharing) |  [link](https://drive.google.com/file/d/1gD8gESbDsgjQjsDnWRJsplDtrGAo59fI/view?usp=sharing) |  [link](https://drive.google.com/file/d/1SMQAD7zBheV84I5BEtgFqCvgDWuXu8CW/view?usp=sharing) | 
	| **`+Un-Mix w/ asymmetric`** | 1000 |  **67.33%**  | [link](https://drive.google.com/file/d/1V9mHEjVgD8FDL_s1Zx6fzouZkzWtlG0o/view?usp=sharing) |  [link](https://drive.google.com/file/d/1VPWKIgu0F8ZYYAFUN4e1pkSQUSDGSBjb/view?usp=sharing) |  [link](https://drive.google.com/file/d/1_Ch9MBKQ4E_Ff3qeLWQZLK8YnDG0HOms/view?usp=sharing) | 
	|   |   |  |  |  |
	| `MoCo w/ symmetric` |  1000 | 65.49% | [link](https://drive.google.com/file/d/1mgCUFSA8Lo38kce97mJcTZ76PQ7INwiC/view?usp=sharing) |  [link](https://drive.google.com/file/d/1LcIh4xmQabzi3oDtwUXHnR_LbspppK-0/view?usp=sharing) | [link](https://drive.google.com/file/d/14bJPtRvD0ZNuOfSAOxc7dCsZg-ejj568/view?usp=sharing) | 
	| **`+Un-Mix w/ symmetric`** | 1000 | **68.83%**  | [link](https://drive.google.com/file/d/1yRVTbJLTL6yphcBWkPw1r29LPcG5stFW/view?usp=sharing) |  [link](https://drive.google.com/file/d/1N3hGWm4kT1x2zrup51avWcxG4V-zte8P/view?usp=sharing) |  [link](https://drive.google.com/file/d/1uwEY7HZUTj1CT6ZfmZ72DR2JSggfaIuh/view?usp=sharing) |  


- Our pre-trained ResNet-50 model on ImageNet (MoCo V2 based) is available at: [Download](https://drive.google.com/file/d/1-t1lgVk6qPBaPRqePPHsEtyf530DfEbt/view?usp=sharing). The training code is available at: [code](https://github.com/szq0214/Un-Mix/tree/master/UnMix_MoCoV2).


## Update (02/15/2021)

We update our [manuscript](https://arxiv.org/pdf/2003.05438.pdf) with a more comprehensive study using *image mixtures* method on unsupervised learning. The core codes of our method can be summarized as follows:

```python:
# P: probability of global or local level mixtures 
# beta: hyperparameter for Beta distribution
# lam: mixture ratio in global-level mixture or bounding box location in region-level mixture
args.beta = 1.0
for x in loader: # load a minibatch x with N samples
    # Probability of choosing global or local level mixtures
    prob = np.random.rand(1)
    lam = np.random.beta(args.beta, args.beta) 
    images_reverse = torch.flip(x[0], (0,))
    if prob < args.P:
	# global-level mixtures
	mixed_images = lam*x[0]+(1-lam)* images_reverse
	mixed_images_flip = torch.flip(mixed_images, (0,))
    else:
	# region-level mixtures
	mixed_images = x[0].clone()
	bbx1, bby1, bbx2, bby2 = utils.rand_bbox(x[0].size(), lam)
	mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images_reverse[:, :, bbx1:bbx2, bby1:bby2] 
	mixed_images_flip = torch.flip(mixed_images,(0,))
	lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x[0].size()[-1] * x[0].size()[-2]))
    # original loss term
    loss_ori = model(x)
    # In following two losses, we found using ''x[0]'' may perform better on some particular datasets
    # loss for the normal order of mixtures
    loss_m1 = model([x[1], mixed_images])
    # loss for the reverse order of mixtures
    loss_m2 = model([x[1], mixed_images_flip])
    # final loss function (our core code)
    loss = loss_ori + lam * loss_m1 + (1-lam) * loss_m2
    # update gradients
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step()
    ...
```

This repo contains the implementation for [Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/2003.05438.pdf), which perturbs input image  space to soften the output prediction space indirectly, meanwhile, assigning new label values in the unsupervised frameworks accordingly. So that the proposed method can smooth decision boundaries and prevent the learner from becoming over-confident.

## Our Motivation: Soften Input/label Spaces

<div align=center>
<img src="./imgs/motivation.png" width="500">
</div>

<div align=left>
Figure 1:  Illustration of our motivation on contrastive-based unsupervised learning approaches. Contrastive learning measures the similarity of sample pairs in the latent representation space. With flattened prediction, the model is encouraged to treat each incorrect instance as equally probable, which will  smooth decision boundaries and prevent the learner from becoming over-confident.
</div> 


## Results

We run our method with [SimCLR](https://arxiv.org/abs/2002.05709?ref=hackernoon.com), [BYOL](https://arxiv.org/abs/2006.07733), [MoCo](https://arxiv.org/abs/1911.05722) and [MoCo V2](https://arxiv.org/abs/2003.04297), the results are as follows:

<div align=center>
<img src="./imgs/res.png" width="800">
</div>

<div align= center>
Figure 2: The curves of training loss and testing accuracy of SimCLR, BYOL and MoCo on CIFAR-10/100 datasets.
</div> 

<div align=center>
<img src="./imgs/res_MoCoV2.png" width="500">
</div>

<div align= center>
Figure 3: Linear classification accuracy of Top-1 (left) and Top-5 (right) with MoCo V2 and ours on ImageNet dataset.
</div> 

<div align=center>
<img src="./imgs/res_ImageNet.png" width="400">
</div>

<div align=center>
Table 1: Comparison of linear classification on standard ImageNet. † denotes our result using multi-scale training.
</div> 

<div align=center>
<img src="./imgs/res_det.png" width="400">
</div>

<div align=center>
Table 2: Object detection results fine-tuned on PASCAL VOC (a) and COCO (b).
</div> 

<!--## An Example of Using this Code on MoCo-->

## Visualizations

<div align=center>
<img src="./imgs/vis1.png" width="600">
</div>

<div align=center>
Figure 4: Visualization of feature embedding with MoCo on CIFAR-10.
</div> 

<div align=center>
<img src="./imgs/vis2.png" width="600">
</div>

<div align=center>
Figure 5: Illustration of weight distributions at 1, 10, 20, 30, 40 and 50-th convolutional layers in a ResNet-50 on ImageNet.
</div> 

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@inproceedings{shen2022unmix,
  title={Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation Learning},
  author={Shen, Zhiqiang and Liu, Zechun and Liu, Zhuang and Savvides, Marios and Darrell, Trevor and Xing, Eric},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2022}
}
```

## Contact

For any questions and comments, please contact Zhiqiang Shen (zhiqiangshen0214 at gmail.com).

## Acknowledgements

MoCo V1&V2 (https://github.com/facebookresearch/moco)

Whitening (https://github.com/htdt/self-supervised)

PyTorch Image Classification (https://github.com/hysts/pytorch_image_classification)
