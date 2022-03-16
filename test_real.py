import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import cv2
import h5py
from makedataset import Dataset

from model import HTDNet, Discriminator
from skimage.measure.simple_metrics import compare_psnr, compare_mse
from skimage.measure import compare_ssim
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim
from loss import *
from torchvision.models import vgg16
import math
from PIL import Image
from perceptual import LossNetwork
#调用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    #开关定义
    parser = argparse.ArgumentParser(description = "network pytorch")
    #train
    parser.add_argument("--epoch", type=int, default = 1000, help = 'epoch number')
    parser.add_argument("--bs", type=str, default =16, help = 'batchsize')
    parser.add_argument("--lr", type=str, default = 1e-4, help = 'learning rate')
    parser.add_argument("--model", type=str, default = "./checkpoint/", help = 'checkpoint')
    #value
    parser.add_argument("--intest", type=str, default = "./input/", help = 'input syn path')
    parser.add_argument("--outest", type=str, default = "./output/", help = 'output syn path')
    argspar = parser.parse_args()
    
    print("\nnetwork pytorch")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    arg = parser.parse_args()
    
    #train
    print('> Loading dataset...')

    FNet, F_optimizer, DNet, D_optimizer, cur_epoch = load_checkpoint(argspar.model, argspar.lr)
    
    test(argspar, FNet)


#加载模型
def load_checkpoint(checkpoint_dir, learnrate):
    Fmodel_name = 'Fmodel.tar'
    Dmodel_name = 'Dmodel.tar'
    if os.path.exists(checkpoint_dir + Fmodel_name):
        #加载存在的模型
        Fmodel_info = torch.load(checkpoint_dir + Fmodel_name)
        Dmodel_info = torch.load(checkpoint_dir + Dmodel_name)
        print('==> loading existing model:', checkpoint_dir + Fmodel_name)
        #模型名称
        FNet = HTDNet()
        DNet = Discriminator()
        #显卡使用
        device_ids = [0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        F_optimizer = torch.optim.Adam(FNet.parameters(), lr=learnrate)
        D_optimizer = torch.optim.Adam(DNet.parameters(), lr=learnrate)

        FNet = torch.nn.DataParallel(FNet, device_ids=device_ids).cuda()
        DNet = torch.nn.DataParallel(DNet, device_ids=device_ids).cuda()
        #将模型参数赋值进net
        FNet.load_state_dict(Fmodel_info['state_dict'])
        F_optimizer = torch.optim.Adam(FNet.parameters())
        F_optimizer.load_state_dict(Fmodel_info['optimizer'])
        DNet.load_state_dict(Dmodel_info['state_dict'])
        D_optimizer = torch.optim.Adam(DNet.parameters())
        D_optimizer.load_state_dict(Dmodel_info['optimizer'])
        cur_epoch = Fmodel_info['epoch']
            
    else:
        # 创建模型
        FNet = HTDNet()
        DNet = Discriminator()
        #显卡使用
        device_ids = [0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        F_optimizer = torch.optim.Adam(FNet.parameters(), lr=learnrate)
        D_optimizer = torch.optim.Adam(DNet.parameters(), lr=learnrate)
        FNet = torch.nn.DataParallel(FNet, device_ids=device_ids).cuda()
        DNet = torch.nn.DataParallel(DNet, device_ids=device_ids).cuda()
        cur_epoch = 0
    return FNet, F_optimizer, DNet, D_optimizer, cur_epoch

def save_checkpoint(stateF, stateD, checkpoint, epoch, mse, psnr, ssim, filename='model.tar'):#保存学习率
    torch.save(stateF, checkpoint + 'Fmodel_%d_%.4f_%.4f_%.4f.tar'%(epoch,mse,psnr,ssim))
    torch.save(stateD, checkpoint + 'Dmodel_%d_%.4f_%.4f_%.4f.tar'%(epoch,mse,psnr,ssim))

#调整学习率
def adjust_learning_rate(optimizer, epoch, lr_update_freq, i):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * i
            print( param_group['lr'])
    return optimizer


def tensor_metric(img, imclean, model, data_range=1):#计算图像PSNR输入为Tensor

    img_cpu = img.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    imgclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    
    SUM = 0
    for i in range(img_cpu.shape[0]):
        if model == 'PSNR':
            SUM += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :],data_range=data_range)
        elif model == 'MSE':
            SUM += compare_mse(imgclean[i, :, :, :], img_cpu[i, :, :, :])
        elif model == 'SSIM':
            SUM += compare_ssim(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range, multichannel = True)
        else:
            print('Model False!')
        
    return SUM/img_cpu.shape[0]

def upsample(x,y):
    _,_,H,W = y.size()
    return F.upsample(x,size=(H,W),mode='bilinear')
  
def test(argspar, model):
    files = os.listdir(argspar.intest) 
    m = 0
    for i in range(len(files)):
        haze = np.array(Image.open(argspar.intest + files[i]))/255  
        model.eval()
        with torch.no_grad():
            
            haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis,:,:,:]).cuda()

            starttime = time.clock()
            T_out, out1, out2, out = model(haze)
            #out1=upsample(out1,T_out)
            #out2=upsample(out2,T_out)
            endtime1 = time.clock()
            m = m + endtime1-starttime
            #torch.cat((haze,T_out,out1, out2,out), dim = 3)
            imwrite(out, argspar.outest+files[i][:-4]+'_DADFNet.png', range=(0, 1))
            #imwrite(out1, argspar.outest+files[i][:-4]+'_our1.png', range=(0, 1))
            #imwrite(out2, argspar.outest+files[i][:-4]+'_our2.png', range=(0, 1))

            print('The '+str(i)+' Time: %.4f.'%(endtime1-starttime))
            print(m)
            


def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] /2#* 0.1
    return optimizer
    
if __name__ == '__main__':
    main()
