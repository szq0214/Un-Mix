# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified by Zhiqiang Shen (http://zhiqiangshen.com/) for Un-Mix on CIFAR10/100 datasets.

import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset
import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV w/ Un-Mix on CIFAR10/100")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/data",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[32], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--use_pil_blur", type=bool_flag, default=True,
                    help="""use PIL library to perform blur instead of opencv""")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--improve_numerical_stability", default=False, type=bool_flag,
                    help="improves numerical stability in Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")

#########################
#### unmix parameters ###
#########################
parser.add_argument("--run_swav", type=bool_flag, default=False,
                    help="whether to train with vanilla swav or not")
parser.add_argument("--unmix_prob", default=0.5, type=float,
                    help="trade-off of local and global mixture")
parser.add_argument("--unmix_beta", default=1.0, type=float,
                    help="beta value in mixture distribution")

def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        pil_blur=args.use_pil_blur,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, queue = train(train_loader, model, optimizer, epoch, lr_schedule, queue)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def swav_loss(embedding, output, args, queue, use_the_queue, reverse):
    # ============ swav loss ... ============
    loss = 0
    bs = args.batch_size
    softmax = nn.Softmax(dim=1).cuda()
    for i, crop_id in enumerate(args.crops_for_assign):
        with torch.no_grad():
            out = output[bs * crop_id: bs * (crop_id + 1)]
            if reverse and crop_id == 0:
                out = torch.flip(out, (0,))

            # time to use the queue
            if queue is not None:
                if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                    use_the_queue = True
                    out = torch.cat((torch.mm(
                        queue[i],
                        model.module.prototypes.weight.t()
                    ), out))
                # fill the queue
                queue[i, bs:] = queue[i, :-bs].clone()
                emb = embedding[crop_id * bs: (crop_id + 1) * bs]
                if reverse and crop_id == 0:
                    emb = torch.flip(emb, (0,))
                queue[i, :bs] = emb
            # get assignments
            q = out / args.epsilon
            if args.improve_numerical_stability:
                M = torch.max(q)
                dist.all_reduce(M, op=dist.ReduceOp.MAX)
                q -= M
            q = torch.exp(q).t()
            q = distributed_sinkhorn(q, args.sinkhorn_iterations)[-bs:]

        # cluster assignment prediction
        subloss = 0
        for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
            p = softmax(output[bs * v: bs * (v + 1)] / args.temperature)
            subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
        loss += subloss / (np.sum(args.nmb_crops) - 1)
    loss /= len(args.crops_for_assign)

    return loss, queue, use_the_queue


def train(train_loader, model, optimizer, epoch, lr_schedule, queue):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, inputs in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        r = np.random.rand(1)
 
        beta = args.unmix_beta
        lam = np.random.beta(beta, beta)
        inputs_unmix = []
        if args.run_swav == False:
            for k in range(0, np.sum(args.nmb_crops), 2):
                images_reverse = torch.flip(inputs[k], (0,))
         
                if r < args.unmix_prob:
                    mixed_images = lam * inputs[k] + (1 - lam) * images_reverse
                else:
                    mixed_images = inputs[k].clone()
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs[k].size(), lam)
                    mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images_reverse[:, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs[k].size()[-1] * inputs[k].size()[-2]))

                inputs_unmix.append(inputs[k])
                inputs_unmix.append(inputs[k+1])
                inputs_unmix.append(mixed_images)

        # ============ multi-res forward passes ... ============
        # =======================================================
        # =======================================================
        # =======================================================
        if args.run_swav == False:
            embedding_unmix, output_unmix = model(inputs_unmix) # multiple crops + mixtures
            embedding_unmix = embedding_unmix.detach()
        else:
            embedding, output = model(inputs) # multiple crops
            embedding = embedding.detach()

        bs = inputs[0].size(0)

        if args.run_swav == False:

            out_unmix = []
            emb_unmix = []
            embedding = []
            output = []
            for crop_id in range(0, len(inputs_unmix), 3):
                out_ori1 = output_unmix[bs * (crop_id + 0): bs * (crop_id + 1)] # one view
                out_ori2 = output_unmix[bs * (crop_id + 1): bs * (crop_id + 2)] # another view
                out_mix = output_unmix[bs * (crop_id + 2): bs * (crop_id + 3)] # mixture

                emb_ori1 = embedding_unmix[bs * (crop_id + 0): bs * (crop_id + 1)] # one view
                emb_ori2 = embedding_unmix[bs * (crop_id + 1): bs * (crop_id + 2)] # another view
                emb_mix = embedding_unmix[bs * (crop_id + 2): bs * (crop_id + 3)]  # mixture

                output.append(torch.cat((out_ori1, out_ori2), dim=0))
                embedding.append(torch.cat((emb_ori1, emb_ori2), dim=0))
                out_unmix.append(torch.cat((out_mix, out_ori2), dim=0))
                emb_unmix.append(torch.cat((emb_mix, emb_ori2), dim=0))

            output = torch.cat(output, dim=0)
            embedding = torch.cat(embedding, dim=0)
            out_unmix = torch.cat(out_unmix, dim=0)
            emb_unmix = torch.cat(emb_unmix, dim=0)

            embedding = embedding.detach()
            emb_unmix = emb_unmix.detach()

        loss = 0

        # ============ swav loss ... ============
        loss_ori, queue, use_the_queue = swav_loss(embedding, output, args, queue, use_the_queue, False)
        if args.run_swav:
            loss = loss_ori
        else:
            loss_unmix, queue, use_the_queue = swav_loss(emb_unmix, out_unmix, args, queue, use_the_queue, False)
            loss_unmix_flip, queue, use_the_queue = swav_loss(emb_unmix, out_unmix, args, queue, use_the_queue, True)

            loss = loss_ori + lam * loss_unmix + (1 - lam) * loss_unmix_flip

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # cancel some gradients
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg), queue


def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (args.world_size * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            dist.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


if __name__ == "__main__":
    main()
