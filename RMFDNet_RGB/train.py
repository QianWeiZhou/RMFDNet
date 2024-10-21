#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import pytorch_ssim
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from apex import amp
from net  import LDF
import os
import cv2
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def rdPrint(anyThing, path='./'):
    orig_stdout = sys.stdout
    f = open(path, 'a+')
    sys.stdout = f

    print(anyThing)

    sys.stdout = orig_stdout
    f.close()
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

def compensation_loss(pred,comp,mask):
    comp = torch.sigmoid(comp)
    pred = torch.sigmoid(pred)
    out1 = pred.clone().detach()

    label = (mask-out1)*mask
    pos = label>0
    pos = pos.float()
    neg = torch.eq(mask, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg
    loss_alpha = F.binary_cross_entropy(comp,label)
    #label = (mask-out1)*mask
    #loss_alpha = loss_alpha/label.sum()
    #label = mask-out1
    #loss_alpha = 1-comp*label
    #loss_alpha[loss_alpha<0]==0
    return loss_alpha,label
def redundant_loss(pred,comp,redt,mask):
    redt = torch.sigmoid(redt)
    #comp = torch.sigmoid(comp)
    pred = torch.sigmoid(pred)
    out1 = pred.clone().detach()
    #com1 = comp.clone().detach()
    label = 1-(mask - out1) * (mask - 1)
    pos = label>0.1
    pos = pos.float()
    neg = torch.eq(mask, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)-num_pos+torch.sum(torch.eq(mask,1).float())
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg
    loss_redund =F.binary_cross_entropy(redt, label)
    #label = (mask-out1)*(mask-1)
    #loss_redund = loss_redund / label.sum()
    return loss_redund,label

def train(Dataset, Network):
    ## dataset
    save_dir = './out/0714'#
    img_dir = save_dir + '/img'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    cfg    = Dataset.Config(datapath='../data/TrainSet/DUTS-TR', savepath=save_dir, mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=60)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=8)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    if not os.path.isdir(cfg.savepath):
        os.mkdir(cfg.savepath)
    os.system('mkdir -p {}/script'.format(os.path.join(cfg.savepath)))  # 要存放当前代码的地方
    os.system('cp -rfp ./*.py {}/script'.format(os.path.join(cfg.savepath)))  # 复制过去
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            #print(name)
            continue
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    #sw             = SummaryWriter(cfg.savepath)
    ## Restore
    # model = ...
    #optimizer = ...
    # checkpoint = torch.load(save_dir+'/model-70')
    epoch1 = 0
    # # # # #net, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)
    # net.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # amp.load_state_dict(checkpoint['amp'])
    # fore_epoch =30
    # behind_epoch = (cfg.epoch-fore_epoch)*2
    for epoch in range(epoch1,cfg.epoch):
        # if epoch<fore_epoch:
        #     optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (fore_epoch*2  + 1) * 2 - 1)) * cfg.lr * 0.1
        #     optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (fore_epoch*2+ 1) * 2 - 1)) * cfg.lr
        # else:
        #     optimizer.param_groups[0]['lr'] = (1 - abs((epoch+cfg.epoch-2*fore_epoch+ 1) / (behind_epoch  + 1) * 2 - 1)) * cfg.lr * 0.1
        #     optimizer.param_groups[1]['lr'] = (1 - abs((epoch+cfg.epoch-2*fore_epoch+ 1) / (behind_epoch + 1) * 2 - 1)) * cfg.lr
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch  + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr
        loss_epoch = 0
        losscomp_epoch = 0
        lossred_epoch = 0
        global_step = 0
        i=0
        for step, (image, mask) in enumerate(loader):
            i=i+1
            image, mask = image.cuda(), mask.cuda()
            #out1, outcom1, outred1, out2, outcom2, outred2= net(image)
            out1, outcom1, outred1, out2, outcom2, outred2, out3, outcom3, outred3, out4, outcom4, outred4, out= net(image)
            # pos = torch.eq(mask, 1).float()
            # neg = torch.eq(mask, 0).float()
            # num_pos = torch.sum(pos)
            # num_neg = torch.sum(neg)
            # num_total = num_pos + num_neg
            #
            # alpha = num_neg / num_total
            # beta = 1.1 * num_pos / num_total
            # # target pixel = 1 -> weight beta
            # # target pixel = 0 -> weight 1-beta
            # weights = alpha * pos + beta * neg

            loss1  = F.binary_cross_entropy_with_logits(out1, mask) + iou_loss(out1, mask)
            com_loss1,_ = compensation_loss(out,outcom1,mask)
            red_loss1,_ = redundant_loss(out,outcom1 ,outred1, mask)
            lossstage1 = loss1  + com_loss1 + red_loss1

            com_loss2,_ = compensation_loss(out,outcom2,mask)
            red_loss2,_ = redundant_loss(out,outcom2, outred2, mask)
            loss2  = F.binary_cross_entropy_with_logits(out2, mask) + iou_loss(out2, mask)
            lossstage2 = loss2 + com_loss2 + red_loss2

            com_loss3,_ = compensation_loss(out,outcom3,mask)
            red_loss3 ,_= redundant_loss(out,outcom3, outred3, mask)
            loss3  = F.binary_cross_entropy_with_logits(out3, mask) + iou_loss(out3, mask)
            lossstage3 = loss3 + com_loss3 + red_loss3

            com_loss4 ,com_label= compensation_loss(out,outcom4,mask)
            red_loss4,red_label = redundant_loss(out,outcom4, outred4, mask)
            loss4  = F.binary_cross_entropy_with_logits(out4, mask) + iou_loss(out4, mask)
            lossstage4 = loss4 + com_loss4 + red_loss4

            out_loss = F.binary_cross_entropy_with_logits(out, mask) + iou_loss(out, mask)
            # com_loss5 ,com_label= compensation_loss(out5,outcom5,mask)
            # red_loss5 ,red_label= redundant_loss(out5,outcom5, outred5, mask)
            # loss5  = F.binary_cross_entropy_with_logits(out5, mask) + iou_loss(out5, mask)
            # lossstage5 = loss5 + com_loss5 + red_loss5
            loss   = (lossstage1+lossstage2+lossstage3+lossstage4)/4+out_loss
            if step ==100 :
                com4 = torch.sigmoid(outcom4[0, 0]).detach().cpu().numpy() * 255
                cv2.imwrite(img_dir + '/' + f'com-{epoch}.png', np.round(com4))
                red4 = torch.sigmoid(outred4[0, 0]).detach().cpu().numpy() * 255
                cv2.imwrite(img_dir + '/' + f'red-{epoch}.png', np.round(red4))
                com_label = com_label[0, 0].detach().cpu().numpy() * 255
                cv2.imwrite(img_dir + '/' + f'compmask-{epoch}.png', np.round(com_label))
                red_label = red_label[0, 0].detach().cpu().numpy() * 255
                cv2.imwrite(img_dir + '/' + f'redmask-{epoch}.png', np.round(red_label))
                out4 = torch.sigmoid(out4[0, 0]).detach().cpu().numpy() * 255
                cv2.imwrite(img_dir + '/' + f'seg-{epoch}.png', np.round(out4))
                pred = torch.sigmoid(out[0, 0]).detach().cpu().numpy() * 255
                cv2.imwrite(img_dir + '/' + f'out-{epoch}.png', np.round(pred))
            #loss = (lossb1 + lossd1 + loss1 + lossb2 + lossd2 + loss2 + com_loss1 + com_loss2 ) / 2
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()
            losscomp_epoch = losscomp_epoch + com_loss4.item()
            lossred_epoch = lossred_epoch + red_loss4.item()
            loss_epoch = loss_epoch+loss4.item()
            ## log
            global_step += 1
           # sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            #sw.add_scalars('loss', {'lossb1':lossb1.item(), 'lossd1':lossd1.item(), 'loss1':loss1.item(), 'lossb2':lossb2.item(), 'lossd2':lossd2.item(), 'loss2':loss2.item()}, global_step=global_step)
            if step%10 == 0:
                #print(out2.max(),out2.min(),out2ma.max(), out2ma.min())
                print('%s | step:%d/%d/%d/%d | lr=%.6f | loss3=%.6f |  loss_comp3=%.6f |loss_red3=%.6f |loss4=%.6f |  loss_comp4=%.6f |loss_red4=%.6f|loss_red4=%.6f'
                    %(datetime.datetime.now(),  i,len(loader), epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'],
                       loss3.item(),  com_loss3.item(),red_loss3.item(),loss4.item(),  com_loss4.item(),red_loss4.item(),out_loss.item()))
            del out1 ,outcom1, outred1, out2,outcom2, outred2, out3, outcom3, outred3, out4,outcom4, outred4,out
            del lossstage1,lossstage2,lossstage3,lossstage4,com_loss1,com_loss2,com_loss3,com_loss4,red_loss1,red_loss2,red_loss3,red_loss4
        rdPrint('%i\t%.7f\t%.7f\t%.7f\t' % (epoch, loss_epoch /i, losscomp_epoch /i, lossred_epoch /i),
                '{}/loss_epoch.log'.format(cfg.savepath))
        #rdPrint('%i\t%.7f\t%.7f\t' % (epoch, loss_epoch /i, losscomp_epoch /i),
        if epoch > (cfg.epoch-10):
            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            }
            torch.save(checkpoint, cfg.savepath+'/model-'+str(epoch+1))



if __name__=='__main__':
    train(dataset, LDF)