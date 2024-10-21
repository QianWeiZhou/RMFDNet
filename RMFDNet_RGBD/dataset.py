#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
from PIL import Image
from PIL import ImageEnhance
########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, depth,mask=None):
        image = (image - self.mean) / self.std
        if mask is None:
            return image,depth/255
        return image, depth/255,mask / 255


class RandomCrop(object):
    def __call__(self, image, depth,mask=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :],depth[p0:p1, p2:p3,:]
        return image[p0:p1, p2:p3, :],depth[p0:p1, p2:p3,:], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, depth,mask=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy(),depth[:, ::-1].copy()
            return image[:, ::-1, :].copy(),depth[:, ::-1,:].copy(),mask[:, ::-1].copy()
        else:
            if mask is None:
                return image,depth
            return image, depth,mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, depth,mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image,depth
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)

        return image, depth,mask


class ToTensor(object):
    def __call__(self, image,depth, mask=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        depth = torch.from_numpy(depth)
        depth = depth.permute(2, 0, 1)
        if mask is None:
            return image,depth
        mask = torch.from_numpy(mask)
        return image,depth, mask

def randomRotation(image,depth,mask):
    if random.random()>0.8:
        imgInfo = image.shape
        height = imgInfo[0]
        width = imgInfo[1]
        random_angle = np.random.randint(-15, 15)
        matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), random_angle, 1)
        image = cv2.warpAffine(image, matRotate, (height, width))
        depth = cv2.warpAffine(depth, matRotate, (height, width))
        mask = cv2.warpAffine(mask, matRotate, (height, width))
    return image,depth,mask
def colorEnhance(image):
    image = Image.fromarray(image)
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return np.array(image)

########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.size = 352
        self.totensor = ToTensor()
        if cfg.mode == 'test':
            self.size = 352
            print(self.size)
        self.resize = Resize(self.size, self.size)

        #self.dataset_list = 'trainNN.txt'
        self.dataset_list = 'train.txt'
        with open(cfg.datapath + '/' + self.dataset_list, 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name = self.samples[idx]
        # print(self.cfg.datapath+'/images/'+name+'.jpg')
        image = cv2.imread(self.cfg.datapath + '/RGB/' + name + '.jpg')#[:, :, ::-1].astype(np.float32)
        print(self.cfg.datapath + '/RGB/' + name + '.jpg')
        #print(image.shape)
        #image = image[:, :, ::-1]
        #print(image.shape)

        if self.cfg.mode == 'train':
            image = colorEnhance(image)
            image = image.astype(np.float32)[:, :, ::-1]
            depth = cv2.imread(self.cfg.datapath + '/depth/' + name + '.bmp')[:, :, ::-1].astype(np.float32)  #
            mask = cv2.imread(self.cfg.datapath + '/GT/' + name + '.png', 0).astype(np.float32)#_GT
            image,depth, mask = self.normalize(image, depth,mask)
            image,depth, mask = self.randomcrop(image,depth, mask)
            image,depth, mask = self.randomflip(image,depth, mask)
            image, depth,mask = randomRotation(image, depth,mask)
            #print(depth.shape)
           #epth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            return image,depth, mask
        else:
            image = image.astype(np.float32)[:, :, ::-1]
            depth = cv2.imread(self.cfg.datapath + '/depth/' + name + '.bmp')#[:, :, ::-1].astype(np.float32)  #
            if depth is None:
                depth = cv2.imread(self.cfg.datapath + '/depth/' + name + '.png')#[:, :, ::-1].astype(np.float32)  #
            shape = image.shape[:2]
            print("此时的name={}".format(name))
            image,depth = self.resize(image,depth)
            image,depth = self.normalize(image,depth)
            image,depth = self.totensor(image,depth)

            return image,depth, shape, name

    def __len__(self):
        return len(self.samples)

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image,depth, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            depth[i] = cv2.resize(depth[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        depth = torch.from_numpy(np.stack(depth, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return image,depth, mask
