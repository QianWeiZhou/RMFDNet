#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import dataset
from torch.utils.data import DataLoader
from net import LDF
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
class Test(object):
    def __init__(self, Dataset, Network, Path):
        ## dataset
        self.epoch = 58
        self.save_dir = f'./'
        self.cfg    = Dataset.Config(datapath=Path, snapshot=f'{self.save_dir}model-{self.epoch}', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            i = 1
            for image, (H, W), name in self.loader:
                if i%300==0:
                    print(f'{i}/{len(self.loader)}')
                i=i+1
                image, shape  = image.cuda().float(), (H, W)
                out1, outcom1, outred1, out2, outcom2, outred2, out3, outcom3, outred3, out4, outcom4, outred4, out = self.net(image, shape)
                #out = out5
                #pred = torch.sigmoid(out[0, 0]).cpu().numpy() * 255
                # #buc = torch.sigmoid(outcom2[0, 0]).cpu().numpy() * 255
                # #duoy= torch.sigmoid(outred2[0, 0]).cpu().numpy() * 255
                head = f'{self.save_dir}'+f'{self.epoch}/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                     os.makedirs(head)
                #cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))
                # com4 = torch.sigmoid(outcom1[0, 0]).cpu().numpy() * 255
                # cv2.imwrite(head + '/' + name[0] + 'comp.png', np.round(com4))
                # red4 = torch.sigmoid(outred1[0, 0]).cpu().numpy() * 255
                # cv2.imwrite(head + '/' + name[0] + 'red.png', np.round(red4))
                # out4 = torch.sigmoid(out4[0, 0]).cpu().numpy() * 255
                # cv2.imwrite(head + '/' + name[0] + 'out4.png', np.round(out4))

                pred = torch.sigmoid(out[0, 0]).cpu().numpy() * 255
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))
                #cv2.imwrite(head + '/' + name[0] + 'c.png', np.round(buc))
                #cv2.imwrite(head + '/' + name[0] + 'r.png', np.round(duoy))
                #if i >50:break
                # pred = torch.sigmoid(outb2[0,0]).cpu().numpy()*255
                # head = f'{self.save_dir}'+'/body-0528'
                # #head = self.cfg.datapath+'/oriModel_body3'
                # if not os.path.exists(head):
                #     os.makedirs(head)
                # cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))
                #
                # pred = torch.sigmoid(outd2[0,0]).cpu().numpy()*255
                # head = f'{self.save_dir}' + '/detail-0528'
                # #head = self.cfg.datapath+'/oriModel_detail3'
                # if not os.path.exists(head):
                #     os.makedirs(head)
                # cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

if __name__=='__main__':
    for path in ['./img']:
        t = Test(dataset, LDF, path)
        t.save()
