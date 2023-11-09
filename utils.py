import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import nibabel as nib
import pandas as pd
from torch.utils import data
import torchvision.models as models
import os


class AE(nn.Module):
    def __init__(self, c):
        self.c = c
        nn.Module.__init__(self)
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, self.c, 3, padding=1), nn.BatchNorm3d(self.c), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.c, 1, 3, padding=1), nn.BatchNorm3d(1), nn.ReLU(True), nn.MaxPool3d(2))
        #Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(1, self.c, 2, stride=2, padding = 0), nn.BatchNorm3d(self.c), nn.ReLU(True),
            nn.ConvTranspose3d(self.c, 3, 2, stride=2, padding = 0), nn.BatchNorm3d(3))
    def forward(self, x):
        x = self.encoder(x)
        encode = x
        x = self.decoder(x)
        return x, encode


class AEup(nn.Module):
    def __init__(self, c):
        nn.Module.__init__(self)
        self.c = c
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, self.c, 3, padding=1), nn.BatchNorm3d(self.c), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.c, 1, 3, padding=1), nn.BatchNorm3d(1), nn.ReLU(True), nn.MaxPool3d(2))
        #Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(size=(20, 20, 20), mode='trilinear'),
            nn.Conv3d(1, self.c, 3, padding=1), nn.BatchNorm3d(self.c), nn.ReLU(True),
            nn.Upsample(size=(40, 40, 40), mode='trilinear'),
            nn.Conv3d(self.c, 3, 3, padding=1), nn.BatchNorm3d(3))
    def forward(self, x):
        x = self.encoder(x)
        encode = x
        x = self.decoder(x)
        return x, encode

class AEcup(nn.Module):
    def __init__(self, c):
        nn.Module.__init__(self)
        self.c = c
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, self.c, 3, padding=1), nn.BatchNorm3d(self.c), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.c, 1, 3, padding=1), nn.BatchNorm3d(1), nn.ReLU(True), nn.MaxPool3d(2))
        #Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(1, self.c, 3, padding=1), nn.BatchNorm3d(self.c), nn.ReLU(True),
            nn.Upsample(size=(20, 20, 20), mode='trilinear'),
            nn.Conv3d(self.c, 3, 3, padding=1), nn.BatchNorm3d(3),
            nn.Upsample(size=(40, 40, 40), mode='trilinear'))
    def forward(self, x):
        x = self.encoder(x)
        encode = x
        x = self.decoder(x)
        return x, encode


class AElin(nn.Module):
    def __init__(self,l):
        nn.Module.__init__(self)
        self.l = l
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 3, 3, padding=1), nn.BatchNorm3d(3), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(3, 1, 3, padding=1), nn.BatchNorm3d(1), nn.ReLU(True), nn.MaxPool3d(2))
        self.linear1 = nn.Sequential(nn.Linear(10* 10* 10, self.l),nn.ReLU(True))
        self.linear2 = nn.Sequential(nn.Linear(self.l, 10* 10* 10),nn.ReLU(True))
        #Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(1, 3, 2, stride=2, padding=0), nn.BatchNorm3d(3), nn.ReLU(True),
            nn.ConvTranspose3d(3, 3, 2, stride=2, padding=0), nn.BatchNorm3d(3))
    def forward(self, x):
        encode = self.encoder(x)
        x = torch.flatten(encode, 1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(-1, 1, 10,10,10)
        x = self.decoder(x)
        return x, encode


class AEconc(nn.Module):
    def __init__(self, c):
        self.c = c
        nn.Module.__init__(self)
        #Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv3d(1, self.c, 3, padding=1), nn.BatchNorm3d(self.c), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.c, 1, 3, padding=1), nn.BatchNorm3d(1), nn.ReLU(True), nn.MaxPool3d(2))
        self.encoder2 = nn.Sequential(
            nn.Conv3d(1, self.c, 3, padding=1), nn.BatchNorm3d(self.c), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.c, 1, 3, padding=1), nn.BatchNorm3d(1), nn.ReLU(True), nn.MaxPool3d(2))
        self.encoder3 = nn.Sequential(
            nn.Conv3d(1, self.c, 3, padding=1), nn.BatchNorm3d(self.c), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(self.c, 1, 3, padding=1), nn.BatchNorm3d(1), nn.ReLU(True), nn.MaxPool3d(2))

        #Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(1, self.c, 2, stride=2, padding = 0), nn.BatchNorm3d(self.c), nn.ReLU(True),
            nn.ConvTranspose3d(self.c, 1, 2, stride=2, padding = 0), nn.BatchNorm3d(1))
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(1, self.c, 2, stride=2, padding = 0), nn.BatchNorm3d(self.c), nn.ReLU(True),
            nn.ConvTranspose3d(self.c, 1, 2, stride=2, padding = 0), nn.BatchNorm3d(1))
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(1, self.c, 2, stride=2, padding = 0), nn.BatchNorm3d(self.c), nn.ReLU(True),
            nn.ConvTranspose3d(self.c, 1, 2, stride=2, padding = 0), nn.BatchNorm3d(1))
    def forward(self, x):
        x1 = self.encoder1(torch.unsqueeze(x[:,0,:,:,:],1))
        x2 = self.encoder2(torch.unsqueeze(x[:,1,:,:,:],1))
        x3 = self.encoder3(torch.unsqueeze(x[:,2,:,:,:],1))
        encode = (x1 + x2 + x3)/3
        xx1 = self.decoder1(encode)
        xx2 = self.decoder2(encode)
        xx3 = self.decoder3(encode)
        return torch.cat((xx1,xx2,xx3),1), encode



class VAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 3, 3, padding=1), nn.BatchNorm3d(3), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(3, 1, 3, padding=1), nn.BatchNorm3d(1), nn.ReLU(True), nn.MaxPool3d(2))
        
        self.encFC1 = nn.Linear(1000, 64)
        self.encFC2 = nn.Linear(1000, 64)
        self.decFC1 = nn.Linear(64, 1000)
        #Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(1, 3, 2, stride=2, padding = 0), nn.BatchNorm3d(3), nn.ReLU(True),
            nn.ConvTranspose3d(3, 3, 2, stride=2, padding = 0), nn.BatchNorm3d(3))
        
    def forward(self, x):
        x = self.encoder(x)
        encode = x
        encodef = encode.view(-1, 10*10*10)
        mu =  self.encFC1(encodef)
        logVar = self.encFC2(encodef)
    
        std = torch.exp(logVar/2)
        z = mu + torch.exp(logVar/2) * torch.randn_like(std)
        x = self.decFC1(z)
        x = x.view(-1, 1, 10, 10, 10)
        x = self.decoder(x)
        return x, mu, logVar,encode




class DatasetFromNii(Dataset):    
    def __init__(self, df):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # read cvs
        self.data_info = df
        # get arrary of image path
        self.a = np.asarray(self.data_info.iloc[:, 0])
        self.data_len = len(self.data_info.index)
    def __getitem__(self, index):
        # read single nii
        aa = np.load(self.a[index]).astype(np.float32) 
        img_as_tensor = torch.from_numpy(aa)
        return img_as_tensor
    def __len__(self):
        return self.data_len
