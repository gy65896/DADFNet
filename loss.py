import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def filter_high_f(fshift,img, radius_ratio):
    """
    过滤掉除了中心区域外的高频信息
    """
    # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        cv.circle(template, (crow, ccol), radius, 1, -1)
    # 2, 过滤掉除了中心区域外的高频信息
    return template * fshift
 
 
def filter_low_f(fshift,img, radius_ratio):
    """
    去除中心区域低频信息
    """
    # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
    filter_img = np.ones(fshift.shape, np.uint8)
    crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv.circle(filter_img, (crow, col), radius, 0, -1)
    # 2 过滤中心低频部分的信息
    return filter_img * fshift
 
 
def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg
 
 
def get_low_high_f(img, radius_ratio):
    """
    获取低频和高频部分图像
    """
    # 傅里叶变换
    # np.fft.fftn
    f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频
 
    # 获取低频和高频部分
    hight_parts_fshift = filter_low_f(fshift.copy(),img, radius_ratio=radius_ratio)  # 过滤掉中心低频
    low_parts_fshift = filter_high_f(fshift.copy(),img, radius_ratio=radius_ratio)
 
    low_parts_img = ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
    high_parts_img = ifft(hight_parts_fshift)
 
    # 显示原始图像和高通滤波处理图像
    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)
 
    # uint8
    img_new_low = np.array(img_new_low*255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high

def fft2torch(image):
    image = 255*image.cpu().detach().numpy()
    low = np.zeros(image.shape)
    high = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            low[i,j,:,:],high[i,j,:,:] = get_low_high_f(image[i,j,:,:], radius_ratio=0.5)

    low = low/255#(low-low.min())/(low.max()-low.min())
    low = torch.from_numpy(low.copy()).type(torch.FloatTensor)
    low = Variable(low.cuda(), requires_grad=True)
    high = high/255#(low-low.min())/(low.max()-low.min())
    high = torch.from_numpy(high.copy()).type(torch.FloatTensor)
    high = Variable(high.cuda(), requires_grad=True)
    return low,high

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])  #算出总共求了多少次差
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个            
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class haze_L1_Loss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(haze_L1_Loss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x,y,z):

        out = torch.abs(((x-y)/torch.clamp((x-z),1e-4))).sum() 
        a,b,c,d = x.shape

        #w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return out/(a*b*c*d)
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
class TV_L1_Loss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TV_L1_Loss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x,y):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])  #算出总共求了多少次差
        count_w = self._tensor_size(x[:,:,:,1:])
        h_xv = (x[:,:,1:,:]-x[:,:,:h_x-1,:])
        h_yv = (y[:,:,1:,:]-y[:,:,:h_x-1,:])

        #h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个            
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_xv = (x[:,:,:,1:]-x[:,:,:,:w_x-1])
        w_yv = (y[:,:,:,1:]-y[:,:,:,:w_x-1])
        L1 = torch.nn.L1Loss()
        #w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return (L1(h_xv,h_yv)+L1(w_xv,w_yv))/2
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

