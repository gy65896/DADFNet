# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:00:46 2020

@author: Administrator
"""

import os
import os.path
import random
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
import argparse
from PIL import Image
class Dataset(udata.Dataset):
	r"""Implements torch.utils.data.Dataset
	"""
	def __init__(self, trainrgb=True,trainsyn = True, shuffle=False):
		super(Dataset, self).__init__()
		self.trainrgb = trainrgb
		self.trainsyn = trainsyn
		self.train_haze	 = 'dataset.h5'
		
		h5f = h5py.File(self.train_haze, 'r')
		  
		self.keys = list(h5f.keys())
		if shuffle:
			random.shuffle(self.keys)
		h5f.close()

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):

		h5f = h5py.File(self.train_haze, 'r')
		  
		key = self.keys[index]
		data = np.array(h5f[key])
		h5f.close()
		return torch.Tensor(data)

def filter_high_f(fshift,img, radius_ratio):
    """
    过滤掉除了中心区域外的高频信息
    """
    # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        cv2.circle(template, (crow, ccol), radius, 1, -1)
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
        cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv2.circle(filter_img, (crow, col), radius, 0, -1)
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
    image = 255*image
    low = np.zeros(image.shape)
    high = np.zeros(image.shape)
    for i in range(image.shape[0]):
        low[i,:,:],high[i,:,:] = get_low_high_f(image[i,:,:], radius_ratio=0.5)

    low = low/255#(low-low.min())/(low.max()-low.min())

    high = high/255#(low-low.min())/(low.max()-low.min())

    return low,high

def data_augmentation(clear ,haze, mode):
    r"""Performs dat augmentation of the input image

    Args:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    clear = np.transpose(clear, (1, 2, 0))
    haze = np.transpose(haze, (1, 2, 0))
    if mode == 0:
        # original
        clear = clear
        haze = haze
    elif mode == 1:
        # flip up and down
        clear = np.flipud(clear)
        haze = np.flipud(haze)
    elif mode == 2:
        # rotate counterwise 90 degree
        clear = np.rot90(clear)
        haze = np.rot90(haze)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        clear = np.rot90(clear)
        clear = np.flipud(clear)
        haze = np.rot90(haze)
        haze = np.flipud(haze)
    elif mode == 4:
        # rotate 180 degree
        clear = np.rot90(clear, k=2)
        haze = np.rot90(haze, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        clear = np.rot90(clear, k=2)
        clear = np.flipud(clear)
        haze = np.rot90(haze, k=2)
        haze = np.flipud(haze)
    elif mode == 6:
        # rotate 270 degree
        clear = np.rot90(clear, k=3)
        haze = np.rot90(haze, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        clear = np.rot90(clear, k=3)
        clear = np.flipud(clear)
        haze = np.rot90(haze, k=3)
        haze = np.flipud(haze)
    else:
        raise Exception('Invalid choice of image transformation')
    return np.transpose(clear, (2, 0, 1)),np.transpose(haze, (2, 0, 1))

def img_to_patches(img,win,stride,Syn=True):
	
	chl,raw,col = img.shape
	chl = int(chl)
	num_raw = np.ceil((raw-win)/stride+1).astype(np.uint8)
	num_col = np.ceil((col-win)/stride+1).astype(np.uint8) 
	count = 0
	total_process = int(num_col)*int(num_raw)
	img_patches = np.zeros([chl,win,win,total_process])
	if Syn:
		for i in range(num_raw):
			for j in range(num_col):			   
				if stride * i + win <= raw and stride * j + win <=col:
					img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, stride*j : stride*j + win]				 
				elif stride * i + win > raw and stride * j + win<=col:
					img_patches[:,:,:,count] = img[:,raw-win : raw,stride * j : stride * j + win]		   
				elif stride * i + win <= raw and stride*j + win>col:
					img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, col-win : col]
				else:
					img_patches[:,:,:,count] = img[:,raw-win : raw,col-win : col]				
				count +=1		   
		
	return img_patches

def normalize(data):

	return np.float32(data/255.)

def samesize(img,size):
	
	img = cv2.resize(img,size)
	return img

def concatenate2imgs(img,depth):	
	c,w,h = img.shape
	conimg = np.zeros((c+1,w,h))
	conimg[0:c,:,:] = img
	conimg[c,:,:] = depth
	
	return conimg

def Train_data():
    '''synthetic Haze images'''
    train_data = 'dataset.h5'
    files1_haze = os.listdir('./input/train/haze1/')
    files1_clear = os.listdir('./input/train/clear1/')
    files2_haze = os.listdir('./input/train/haze2/')
    files2_clear = os.listdir('./input/train/clear2/')
    	  

    with h5py.File(train_data, 'w') as h5f:
        count1 = 0

        
        for i in range(len(files1_haze)):
            haze1 = np.array(Image.open('./input/train/haze1/' + files1_haze[i]))/255
            clear1 = np.array(Image.open('./input/train/clear1/' + files1_clear[i]))/255

            haze1 = haze1.transpose(2, 0, 1)
            clear1 = clear1.transpose(2, 0, 1)
            haze1 = img_to_patches(haze1,256,200)
            clear1 = img_to_patches(clear1,256,200)
            for nx in range(clear1.shape[3]):
                haze,clear = data_augmentation(haze1[:, :, :, nx].copy(),clear1[:, :, :, nx].copy(), np.random.randint(0, 7))
                clear_high,clear_low = fft2torch(clear)
                # print(clear_high.shape,clear_low.shape)
                # cv2.imwrite('./1.png',np.transpose(np.concatenate([haze,clear,clear_high,clear_low],2), (1, 2, 0))*255)
                data1 = np.concatenate([haze,clear_high,clear_low,clear],0)
                h5f.create_dataset(str(count1), data=data1)
                count1 += 1
                print(count1,clear1.shape[3],data1.shape)  
        count2 = 0
        for i in range(len(files2_haze)):
            haze2 = np.array(Image.open('./input/train/haze2/' + files2_haze[i]))/255
            clear2 = np.array(Image.open('./input/train/clear2/' + files2_clear[i]))/255

            haze2 = haze2.transpose(2, 0, 1)
            clear2 = clear2.transpose(2, 0, 1)
            haze2 = img_to_patches(haze2,256,256)
            clear2 = img_to_patches(clear2,256,256)
            for nx in range(clear2.shape[3]):
                haze,clear = data_augmentation(haze2[:, :, :, nx].copy(),clear2[:, :, :, nx].copy(), np.random.randint(0, 7))
                clear_high,clear_low = fft2torch(clear)
                
                data2 = np.concatenate([haze,clear_high,clear_low,clear],0)
                h5f.create_dataset(str(count1), data=data2)
                count1 += 1
                count2 += 1
                print(count2,clear2.shape[3],data2.shape) 
        print(count1-count2,count2) 

    h5f.close()	

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description = "Building the training patch database")
    parser.add_argument("--patch_size", "--p", type = int, default=128, help="Patch size")
    parser.add_argument("--stride", "--s", type = int, default=64, help="Size of stride")
    args = parser.parse_args()
    
    Train_data()