import numpy as np
import random
import torch
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
from .utils import rand_bbox

class GM(Dataset):
    def __init__(self, dataset, alpha, num_mix=2, prob=0.5):
        self.dataset = dataset
        self.alpha = alpha
        self.num_mix = num_mix
        self.prob = prob

    def __getitem__(self, index):
        data = self.dataset[index]

        # global level mixture
        for _ in range(self.num_mix):
            p = np.random.rand(1)
            if p > self.prob:
                continue

            # generate mixed sample
            rand_index = random.choice(range(len(self)))
            data2 = self.dataset[rand_index]
            lam = np.random.beta(self.alpha, self.alpha)
            data[0][:, :, :] = data[0][:, :, :] * lam + data2[0][:, :, :] * (1 - lam)

        return data

    def __len__(self):
        return len(self.dataset)

class RM(Dataset):
    def __init__(self, dataset, num_mix=1, beta=1., prob=1.0, decay=1.0): 
        self.dataset = dataset
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.decay = decay

    def __getitem__(self, index):
        data = self.dataset[index]
        
        # region level mixture
        for mix_index in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            data2 = self.dataset[rand_index]

            bbx1, bby1, bbx2, bby2 = rand_bbox(data[0].size(), lam)
            if mix_index == 0:  #pixel decay
                data = list(data)
                data[0] = self.decay * data[0]
                data = tuple(data)
            try:
                data[0][:, bbx1:bbx2, bby1:bby2] = data2[0][:, bbx1:bbx2, bby1:bby2]
            except:
                print(bbx1, bby1, bbx2, bby2)
                continue

        return data

    def __len__(self):
        return len(self.dataset)

class Cutout(Dataset):
    def __init__(self, dataset, mask_size, p, cutout_inside, mask_color=0):
        self.dataset = dataset
        self.p = p
        self.mask_size = mask_size
        self.cutout_inside = cutout_inside
        self.mask_color = mask_color

        self.mask_size_half = mask_size // 2
        self.offset = 1 if mask_size % 2 == 0 else 0

    def __getitem__(self, index):
        data = self.dataset[index]
        # image = np.asarray(data).copy()

        if np.random.random() > self.p:
            return data

        h, w = data[0].shape[1:]

        if self.cutout_inside:
            cxmin, cxmax = self.mask_size_half, w + self.offset - self.mask_size_half
            cymin, cymax = self.mask_size_half, h + self.offset - self.mask_size_half
        else:
            cxmin, cxmax = 0, w + self.offset
            cymin, cymax = 0, h + self.offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - self.mask_size_half
        ymin = cy - self.mask_size_half
        xmax = xmin + self.mask_size
        ymax = ymin + self.mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        data[0][:, xmin:xmax, ymin:ymax] = self.mask_color
        return data

    def __len__(self):
        return len(self.dataset)

class RandomErasing(Dataset):
    def __init__(self, dataset, p, area_ratio_range, min_aspect_ratio, max_attempt):
        self.dataset = dataset
        self.p = p
        self.max_attempt = max_attempt
        self.sl, self.sh = area_ratio_range
        self.rl, self.rh = min_aspect_ratio, 1. / min_aspect_ratio

    def __getitem__(self, index):
        data = self.dataset[index]
        # image = np.asarray(data).copy()

        if np.random.random() > self.p:
            return data

        h, w = data[0].shape[1:]
        image_area = h * w

        for _ in range(self.max_attempt):
            mask_area = np.random.uniform(self.sl, self.sh) * image_area
            aspect_ratio = np.random.uniform(self.rl, self.rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                data[0][:, x0:x1, y0:y1] = np.random.uniform(0, 1)
                break

        return data

    def __len__(self):
        return len(self.dataset)

def IM(train_dataset, g_alpha, g_num_mix, g_prob, r_beta, r_prob, r_num_mix, r_decay):
    train_dataset = GM(train_dataset, g_alpha, g_num_mix, g_prob)
    train_dataset = RM(train_dataset, r_beta, r_prob, r_num_mix, r_decay)
    return train_dataset

def global_(train_dataset, g_alpha, g_num_mix, g_prob):
    train_dataset = GM(train_dataset, g_alpha, g_num_mix, g_prob)
    return train_dataset

def region(train_dataset, r_beta, r_prob, r_num_mix, r_decay):
    train_dataset = RM(train_dataset, r_beta, r_prob, r_num_mix, r_decay)
    return train_dataset
