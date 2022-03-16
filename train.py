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
import pandas as pd
#调用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    #开关定义
    parser = argparse.ArgumentParser(description = "network pytorch")
    #train
    parser.add_argument("--epoch", type=int, default = 200, help = 'epoch number')
    parser.add_argument("--bs", type=str, default =4, help = 'batchsize')
    parser.add_argument("--lr", type=str, default = 1e-4, help = 'learning rate')
    parser.add_argument("--model", type=str, default = "./checkpoint/", help = 'checkpoint')
    #value
    parser.add_argument("--intest", type=str, default = "./input/test/", help = 'input syn path')
    parser.add_argument("--outest", type=str, default = "./output/value/", help = 'output syn path')
    argspar = parser.parse_args()
    
    print("\nnetwork pytorch")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    arg = parser.parse_args()
    
    #train
    print('> Loading dataset...')
    data = Dataset(shuffle=True)
    dataset = DataLoader(dataset=data, num_workers=0, batch_size=argspar.bs, shuffle=True)
    FNet, F_optimizer, DNet, D_optimizer, cur_epoch = load_checkpoint(argspar.model, argspar.lr)
    
    start_all = time.clock()
    train(FNet, F_optimizer, DNet, D_optimizer, cur_epoch, arg, dataset)   
    end_all = time.clock()
    
    print('Whloe Training Time:' +str(end_all-start_all)+'s.')

def load_excel(x,y,z,m):
    data1 = pd.DataFrame(x)
    data2 = pd.DataFrame(y)
    data3 = pd.DataFrame(z)
    data4 = pd.DataFrame(m)
    writer = pd.ExcelWriter('./A.xlsx')		# 写入Excel文件
    data1.to_excel(writer, 'SOTS-PSNR', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    data2.to_excel(writer, 'SOTS-SSIM', float_format='%.5f')
    data3.to_excel(writer, 'NH-PSNR', float_format='%.5f')
    data4.to_excel(writer, 'NH-SSIM', float_format='%.5f')
    writer.save()
    writer.close()

def train(FNet, F_optimizer, DNet, D_optimizer, cur_epoch, argspar, dataset):
    #loss
    vgg_model = vgg16(pretrained=True)
    vgg_model = vgg_model.features[:16].cuda()
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    L2_loss = torch.nn.MSELoss().cuda()
    L1_loss = torch.nn.L1Loss().cuda()
    L1_loss_2 = haze_L1_Loss().cuda()
    msssim_loss = msssim
    adversarial_loss = nn.BCEWithLogitsLoss().cuda()
    psnr_all1 = []
    ssim_all1 = []
    psnr_all2 = []
    ssim_all2 = []
    #train
    for epoch in range(cur_epoch, argspar.epoch):
        F_optimizer = adjust_learning_rate(F_optimizer, epoch, 30)
        D_optimizer = adjust_learning_rate(D_optimizer, epoch, 30)
        #optimizer = adjust_learning_rate(optimizer, epoch, argspar.lr, 0.5)
        
        learnrate = F_optimizer.param_groups[-1]['lr']

        FNet.train()
        DNet.train()
        #合成图像训练

        for i,data in enumerate(dataset,0):
            haze,clear_low,clear_high,clear = data[:,:3,:,:].cuda(),data[:,3:6,:,:].cuda(),\
                data[:,6:9,:,:].cuda(),data[:,-3:,:,:].cuda()


            #模型输出
            A_map, out1, out2, out = FNet(haze)
            
            DNet.zero_grad()
            real_out = DNet(clear).mean()
            #fake_out1 = DNet(out1).mean()
            #fake_out2 = DNet(out2).mean()
            fake_out = DNet(out).mean()
            
            D_loss = adversarial_loss(fake_out, torch.zeros_like(fake_out)) + \
            adversarial_loss(real_out, torch.ones_like(real_out))#
            D_loss = 1 - real_out + fake_out
            D_loss.backward(retain_graph=True)
            
            FNet.zero_grad()
            perceptual_loss = loss_network(out, clear)
            a_loss = torch.mean(1 - fake_out)#adversarial_loss(fake_out, torch.ones_like(fake_out))#torch.mean(1 - fake_out)

            clear_low = maxpool(maxpool(clear_low))
            clear_high = maxpool(maxpool(clear_high))#L1_loss_2(out, clear,haze)
            smooth_loss_l1 = F.smooth_l1_loss(out, clear)+0.5*(F.smooth_l1_loss(out1, clear_low)+\
                                    F.smooth_l1_loss(out2, clear_high))#+0.5*F.smooth_l1_loss(out2_high, clear2_high))
            msssim_loss_ = 1-msssim_loss(out, clear, normalize=True)
            
            total_loss = smooth_loss_l1 + 0.01*perceptual_loss+ 0.5*msssim_loss_ + 0.0005*a_loss#+ 0.5*msssim_loss_
            #imwrite(torch.cat((clear2,clear2_low,clear2_high,out1,out2), dim = 2), './1.png', range=(0, 1))
            total_loss.backward()
            D_optimizer.step()
            F_optimizer.step()

            mse = tensor_metric(clear,out, 'MSE', data_range=1)
            psnr = tensor_metric(clear,out, 'PSNR', data_range=1)
            ssim = tensor_metric(clear,out, 'SSIM', data_range=1)
            print("[epoch %d][%d/%d] lr :%f Floss: %.4f Dloss: %.4f fake: %.4f real: %.4f MSE: %.4f PSNR: %.4f SSIM: %.4f"%(epoch+1, i+1, \
                len(dataset), learnrate, total_loss.item(),D_loss.item(),fake_out,real_out, mse, psnr, ssim))

            

        psnr_t1,ssim_t1,psnr_t2,ssim_t2 = test(argspar, FNet, epoch)
        psnr_all1.append(psnr_t1)
        ssim_all1.append(ssim_t1)
        psnr_all2.append(psnr_t2)
        ssim_all2.append(ssim_t2)
        print("[epoch %d] Test images PSNR1: %.4f SSIM1: %.4f PSNR2: %.4f SSIM2: %.4f"%(epoch+1, psnr_t1,ssim_t1,psnr_t2,ssim_t2))
        load_excel(psnr_all1,ssim_all1,psnr_all2,ssim_all2)
        #保存模型
        save_checkpoint({'epoch': epoch + 1,'state_dict': FNet.state_dict(),'optimizer' : F_optimizer.state_dict()},\
                        {'epoch': epoch + 1,'state_dict': DNet.state_dict(),'optimizer' : D_optimizer.state_dict()}, \
                        argspar.model, epoch+1,psnr_t1,ssim_t1,psnr_t2,ssim_t2)


#加载模型
def load_checkpoint(checkpoint_dir, learnrate):
    Fmodel_name = 'Fmodel_77_25.7723_0.9564_19.4342_0.8069.tar'
    Dmodel_name = 'Dmodel_77_25.7723_0.9564_19.4342_0.8069.tar'
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

def save_checkpoint(stateF, stateD, checkpoint, epoch, psnr1, ssim1, psnr2, ssim2, filename='model.tar'):#保存学习率
    torch.save(stateF, checkpoint + 'Fmodel_%d_%.4f_%.4f_%.4f_%.4f.tar'%(epoch,psnr1, ssim1, psnr2, ssim2))
    torch.save(stateD, checkpoint + 'Dmodel_%d_%.4f_%.4f_%.4f_%.4f.tar'%(epoch,psnr1, ssim1, psnr2, ssim2))

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
  
def test(argspar, model, epoch = -1):
    files1 = os.listdir(argspar.intest + 'clear1/') 
    psnr1, ssim1 = 0, 0
    for i in range(len(files1)):
        clear = np.array(Image.open(argspar.intest + 'clear1/' + files1[i]))/255
        haze = np.array(Image.open(argspar.intest + 'haze1/' + files1[i]))/255  
        model.eval()
        with torch.no_grad():
            haze,clear = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis,:,:,:]).cuda(),torch.Tensor(clear.transpose(2, 0, 1)[np.newaxis,:,:,:]).cuda()

            starttime = time.clock()
            T_out, out1, out2, out = model(haze)
            endtime1 = time.clock()
            #clear2_low,clear2_high = fft2torch(clear)

            out1=upsample(out1,T_out)
            out2=upsample(out2,T_out)
            
            
            imwrite(torch.cat((clear,haze,T_out,out1,out2,out), dim = 3), argspar.outest+str(i)+'_'+str(epoch)+'.png', range=(0, 1))
            psnr1 += tensor_metric(clear,out, 'PSNR', data_range=1)
            ssim1 += tensor_metric(clear,out, 'SSIM', data_range=1)
            print('The '+str(i)+' Time:' +str(endtime1-starttime)+'s.')
    files2 = os.listdir(argspar.intest + 'clear2/') 
    psnr2, ssim2 = 0, 0
    for i in range(len(files2)):
        clear = np.array(Image.open(argspar.intest + 'clear2/' + files2[i]))/255
        haze = np.array(Image.open(argspar.intest + 'haze2/' + files2[i]))/255  
        model.eval()
        with torch.no_grad():
            haze,clear = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis,:,:,:]).cuda(),torch.Tensor(clear.transpose(2, 0, 1)[np.newaxis,:,:,:]).cuda()

            starttime = time.clock()
            T_out, out1, out2, out = model(haze)
            endtime1 = time.clock()
            #clear2_low,clear2_high = fft2torch(clear)

            out1=upsample(out1,T_out)
            out2=upsample(out2,T_out)
            
            
            imwrite(torch.cat((clear,haze,T_out,out1,out2,out), dim = 3), argspar.outest+str(i+len(files1))+'_'+str(epoch)+'.png', range=(0, 1))
            psnr2 += tensor_metric(clear,out, 'PSNR', data_range=1)
            ssim2 += tensor_metric(clear,out, 'SSIM', data_range=1)
            print('The '+str(i)+' Time:' +str(endtime1-starttime)+'s.')
    return psnr1/len(files1), ssim1/len(files1),psnr2/len(files2), ssim2/len(files2)
            


def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] /2#* 0.1
    return optimizer
    
if __name__ == '__main__':
    main()
