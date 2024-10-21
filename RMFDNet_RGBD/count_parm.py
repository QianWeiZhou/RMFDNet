import torch
import torch.nn as nn
from net import RCRNet
#import dataset
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#model = LDF()
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
#cfg    = Dataset.Config(datapath=Path, snapshot=f'{self.save_dir}model-{self.epoch}',mode='test')
model = RCRNet()
#total conv parameters: 30.454593M total conv GFLOPs: 152.380467200
#total conv parameters: 47.746434M total conv GFLOPs:  29.152697600
#total conv parameters: 43.981069M total conv GFLOPs: 117.196492800
#total conv parameters: 34.882310M total conv GFLOPs: 574.452531200
#model = XXX_Net()#注：以下代码放在模型实例化之后，模型名用model
def my_hook(Module, input, output):
    outshapes.append(output.shape)
    modules.append(Module)

names,modules,outshapes = [],[],[]
for name,m in model.named_modules():
    if isinstance(m,nn.Conv2d):
        m.register_forward_hook(my_hook)
        names.append(name)

def calc_paras_flops(modules,outshapes):
    total_para_nums = 0
    total_flops = 0
    for i,m in enumerate(modules):
        Cin = m.in_channels
        Cout = m.out_channels
        k = m.kernel_size
        #p = m.padding
        #s = m.stride
        #d = m.dilation
        g = m.groups
        Hout = outshapes[i][2]
        Wout = outshapes[i][3]
        if m.bias is None:
            para_nums = k[0] * k[1] * Cin / g * Cout
            flops = (2 * k[0] * k[1] * Cin/g - 1) * Cout * Hout * Wout
        else:
            para_nums = (k[0] * k[1] * Cin / g +1) * Cout
            flops = 2 * k[0] * k[1] * Cin/g * Cout * Hout * Wout
        para_nums = int(para_nums)
        flops = int(flops)
        #print(names[i], 'para:', para_nums, 'flops:',flops)
        total_para_nums += para_nums
        total_flops += flops
    print('total conv parameters:',total_para_nums, 'total conv FLOPs:',total_flops)
    return total_para_nums, total_flops

input = torch.rand(1,3,352,352)#需要先提供一个输入张量
depth  = torch.rand(1,3,352,352)
input = torch.FloatTensor(input)
depth = torch.FloatTensor(depth)

out1= model(input,depth)
total_para_nums, total_flops = calc_paras_flops(modules,outshapes)
