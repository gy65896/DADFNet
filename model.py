# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:28:51 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import numpy as np
import math
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=3,channel =16):
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(channel,in_planes,kernel_size=1,stride=1,padding=0),nn.Sigmoid())

    def forward(self, x):
        out = self.conv2(self.avg_pool(self.conv1(x)))
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, in_planes=3, channel=16, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_planes,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=3, dilation=3)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=5, dilation=5)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=7, dilation=7)
        self.conv4 = nn.Sequential(
            nn.Conv2d(4,1,kernel_size=3,stride=1,padding=1),nn.Sigmoid())

    def forward(self, x):
        x    = self.conv0(x)
        x1_1 = self.conv1(x)
        x1_2 = self.conv2(x)
        x1_3 = self.conv3(x)
        
        avg_x = torch.mean(x, dim=1, keepdim=True)
        avg_x1 = torch.mean(x1_1, dim=1, keepdim=True)
        avg_x2 = torch.mean(x1_2, dim=1, keepdim=True)
        avg_x3 = torch.mean(x1_3, dim=1, keepdim=True)
        
        out = torch.cat([avg_x1, avg_x2, avg_x3, avg_x], dim=1)
        out = self.conv4(out)
        return out

class HTDNet(nn.Module):
    def __init__(self):
        super(HTDNet,self).__init__()
        self.a = T_Attention()
        self.t = TIDFE()
        self.relu = nn.LeakyReLU()

    def forward(self,x):
        A_map = self.a(x)	
        x_in = A_map+x
        I_out, F_out, out = self.t(x_in)
        out = self.relu(out)
        return A_map, I_out, F_out, out
    
class T_Attention(nn.Module):
    def __init__(self,channel_in=3):
        super(T_Attention,self).__init__()
        self.CA = ChannelAttention()
        self.MSSA = SpatialAttention()
    def forward(self,x):
        x1 = self.CA(x)*x
        x2 = self.MSSA(x1)*x1
        
        return x2

class TIDFE(nn.Module):
    def __init__(self, channel_in = 3, channel_out = 3, channel_1 = 16, channel_2 = 32, channel_3 = 64):
        super(TIDFE, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(channel_in,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(channel_1,channel_2,kernel_size=3,stride=2,padding=1),nn.LeakyReLU())
        
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(channel_1,channel_2,kernel_size=3,stride=2,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(channel_2,channel_3,kernel_size=3,stride=2,padding=1),nn.LeakyReLU())
        
        self.I = I_Net(channel_3)
        
        self.F = F_Net(channel_3)
        
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(channel_3,channel_out,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(channel_3,channel_out,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())   
        
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(channel_3,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_3,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_3,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_3,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_3,channel_3,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        
        self.conv4_0 = nn.Sequential(
            nn.Conv2d(channel_3*3,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_2,channel_2,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        
        self.conv5_0 = nn.Sequential(
            nn.Conv2d(channel_2,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
            nn.Conv2d(channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(channel_1,channel_out,kernel_size=3,stride=1,padding=1))


        # self.conv_6_1 = nn.Sequential(
        #     nn.Conv2d(channel_in,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
        #     nn.Conv2d(channel_1,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        # self.conv_6_2 = nn.Sequential(
        #     nn.Conv2d(channel_1*2,channel_1,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
        #     nn.Conv2d(channel_1,channel_out,kernel_size=3,stride=1,padding=1),nn.LeakyReLU(),
        #     nn.AdaptiveAvgPool2d(1))
        # self.conv_6_3 = nn.Sequential(
        #     nn.Conv2d(channel_out,channel_out,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())

        # self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    def _upsample(self,x,y):
        _,_,H,W = y.size()
        return F.upsample(x,size=(H,W),mode='bilinear')
    def forward(self,x):
        #encoder
        x1_1 = self.conv1_1(x)
        x2_1 = self.conv2_1(x1_1)
        x3_1 = self.conv3_1(x2_1+self.conv1_2(x1_1))
        #middle
        x3_2 = self.I(x3_1)
        x3_3 = self.F(x3_1)
        x3_4 = self.conv3_4(x3_1)
        I_out = self.conv3_2(x3_2)
        F_out = self.conv3_3(x3_3)
        #decoder
        x4_0 = self.conv4_0(self._upsample(torch.cat([x3_4,x3_2,x3_3],1),x2_1))
        x4_1 = self.conv4_1(x4_0+x2_1)
        x5_0 = self.conv5_0(self._upsample(x4_1+x4_0,x1_1))
        x5_1 = self.conv5_1(x5_0+x1_1)
        
        out = self.conv5_2(x5_1)
        
        # x6_2 = self.conv_6_1(x)
        # print(k[0,0,:,:])
        # x6_2 = torch.cat([x6_2,x5_1],1)
        # b = self.conv_6_2(x6_2)
        

        return I_out, F_out, out
class I_Net(nn.Module):
    def __init__(self,channel):
        super(I_Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=2, dilation=2),nn.LeakyReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=4, dilation=4),nn.LeakyReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv6 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=2, dilation=2),nn.LeakyReLU())
        self.conv7 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1))
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        x1_1 = self.conv1(x)
        x1_2 = self.conv2(x1_1)
        x1_3 = self.conv3(x1_1+x1_2)  	   
        x1_4 = self.conv4(x1_3) 
        x1_5 = self.conv5(x1_3+x1_4) 
        x1_6 = self.conv6(x1_5)
        x_out = self.relu(self.conv7(x1_6)+x)
        return x_out

class F_Net(nn.Module):
    def __init__(self, channel):
        super(F_Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())       
        self.conv6 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
        self.conv7 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),nn.LeakyReLU())
    def forward(self,x):
        x1_1 = self.conv1(x)
        x1_2 = self.conv2(x1_1)  
        x1_3 = self.conv3(x1_2) 	  
        x1_4 = self.conv4(x1_2+x1_3)  
        x1_5 = self.conv5(x1_2+x1_3+x1_4)  
        x1_6 = self.conv6(x1_2+x1_3+x1_4+x1_5)   
        x_out = self.conv7(x1_6)	
        return x_out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 , 16, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(16),nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(32),nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(32),nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3),nn.BatchNorm2d(64),nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=5, dilation=5),nn.BatchNorm2d(64),nn.LeakyReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=7, dilation=7),nn.BatchNorm2d(64),nn.LeakyReLU())
        self.net2 = nn.Sequential(
            nn.Conv2d(64*4, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)  # ,

            # nn.Sigmoid()
        )

    def forward(self, x):
        x_1 = self.net(x)
        x_2 = self.conv1(x_1)
        x_3 = self.conv2(x_1)
        x_4 = self.conv3(x_1)
        out = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        
        batch_size = x.size(0)
        return torch.sigmoid(self.net2(out).view(batch_size))
