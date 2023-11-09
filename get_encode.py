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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from utils import *
from nilearn.image import resample_img 
from scipy.ndimage import zoom
from nilearn import plotting
import time
from sklearn.model_selection import KFold
import pickle
from skimage.metrics import structural_similarity 
from skimage.metrics import peak_signal_noise_ratio 
import pathlib
import sys 


data = [x for x in range(0,502)]
kfold = KFold(5)
fold_id =[]
for train, test in kfold.split(data):
    fold_id.append([train,test])

def get_test(data, f):
    l = []
    for i in os.listdir("/data/" + data +"/"):
        id_ = int(i.split("_")[0][3:])
        if id_ in fold_id[f][1]:
            l.append("/data/" + data +"/" +i)
    df = pd.DataFrame(l)
    return df

 

class DatasetFromNii(Dataset):    
    def __init__(self, df):
        self.to_tensor = transforms.ToTensor()
        self.data_info = df
        self.a = np.asarray(self.data_info.iloc[:, 0])
        self.data_len = len(self.data_info.index)
    def __getitem__(self, index):
        # read single nii
        aa = np.load(self.a[index]).astype(np.float32) 
        img_as_tensor = torch.from_numpy(aa)
        return img_as_tensor

    def __len__(self):
        return self.data_len

### get ReconstrMSE for model
def get_model(model_name):
    if model_name == "AE3":
        model = AE(3)
    elif model_name == "AElin64":
        model = AElin(64)
    elif model_name == "AEup3":
        model = AEup(3)
    elif model_name == "AEconc3":
        model = AEconc(3)
    return model

def get_mse(df, model_pth):
    print("#",model_name,len(df))
    model = get_model(model_name) 
    model.load_state_dict(torch.load(model_pth, map_location='cpu'))
    model.eval()
    
    all_mse = 0
    all_psnr = 0
    all_ssim = 0
    for i, r in df.iterrows():
        a = np.load(r[0])
        aa = torch.from_numpy(np.expand_dims(a,axis = 0).astype(np.float32))
        out, encode = model(aa)
        out = out[0,:,:,:,:].detach().cpu().numpy()
        af = a.flatten()
        bf = out.flatten()
        mse = ((af - bf)**2).mean(axis=0)
        
        max_value = np.max(af) if np.max(af) > np.max(bf) else np.max(bf)
        psnr = 20 * np.log10(max_value / (np.sqrt(mse)))
        
 
        ssim = structural_similarity(af, bf)
        
        all_psnr += psnr
        all_mse += mse
        all_ssim += ssim
        
    all_mse = all_mse/len(df)
    all_psnr = all_psnr/len(df)
    all_ssim = all_ssim/len(df)

    return np.round(all_mse,3), np.round(all_psnr,3), np.round(all_ssim,3)

def get_encode(ind_, data_type):
    lpatch = 40
    a = nib.load("/LRP/LRP/All_maps/t1/modelB44/" + data_type + "_map" + str(ind_) + ".nii.gz").get_fdata()
    b = nib.load("/LRP/IG/All_maps/t1/modelB44/" + data_type + "_map" + str(ind_) + ".nii.gz").get_fdata()
    c = nib.load("/LRP/GGC/All_maps/t1/modelB44/" + data_type + "_map" + str(ind_) + ".nii.gz").get_fdata()
    a = np.pad(a, ((0,7), (0,11), (0,7)), 'constant')
    b = np.pad(b, ((0,7), (0,11), (0,7)), 'constant')
    c = np.pad(c, ((0,7), (0,11), (0,7)), 'constant')

    encode_map = np.zeros((10*5, 10*6,10*5))
    out_map = np.zeros((3, 40*5, 40*6,40*5))
    for x in range(0,5):
        for y in range(0,6):
            for z in range(0,5):
                allmap = np.expand_dims(np.concatenate([
                    np.expand_dims(a[x*lpatch:x*lpatch + lpatch, y*lpatch:y*lpatch + lpatch, z*lpatch:z*lpatch + lpatch],axis =0),
                    np.expand_dims(b[x*lpatch:x*lpatch + lpatch, y*lpatch:y*lpatch + lpatch, z*lpatch:z*lpatch + lpatch],axis =0),
                    np.expand_dims(c[x*lpatch:x*lpatch + lpatch, y*lpatch:y*lpatch + lpatch,z*lpatch:z*lpatch + lpatch],axis =0)], 
                    axis=0),axis=0)
                allmap = torch.from_numpy(allmap.astype(np.float32))
                out, encode = model(allmap)

                encode_map[x*10:x*10 + 10, y*10:y*10 + 10, z*10:z*10 + 10] = encode[0,0,:,:,:].detach().numpy()
                out_map[0, x*40:x*40 + 40, y*40:y*40 + 40, z*40:z*40 + 40] = out[0,0,:,:,:].detach().numpy()
                out_map[1, x*40:x*40 + 40, y*40:y*40 + 40, z*40:z*40 + 40] = out[0,1,:,:,:].detach().numpy()
                out_map[2, x*40:x*40 + 40, y*40:y*40 + 40, z*40:z*40 + 40] = out[0,2,:,:,:].detach().numpy()
    return encode_map, out_map[:, 0:193,0:229,0:193]



data_type = "zabs"
folder_name = sys.argv[1]
model_name = folder_name.split("_")[0]

print(model_name)
for i in range(0,5):
    model_pth = "/results/patch3D_40_zabs/" + folder_name + "/model_f" + str(i) + ".pth"
    model = get_model(model_name) 
    model.load_state_dict(torch.load(model_pth, map_location='cpu'))
    model.eval()

    encode_map = np.zeros((10*5, 10*6,10*5))
    for s in fold_id[i][1]:
        encode_map,_ = get_encode(s, data_type)
        encode_map += encode_map
    encode_map = encode_map /len(fold_id[i][1])
    np.save("/results/patch3D_40_zabs/" + folder_name + "/encode_map_f" + str(i) +".npy", encode_map)
    
encode_map = np.zeros((10*5, 10*6,10*5))
for i in range(0,5):
    encode_map =  np.load("/results/patch3D_40_zabs/" + folder_name + "/encode_map_f" + str(i) +".npy")
    encode_map += encode_map
encode_map = encode_map/5    
np.save("/results/patch3D_40_zabs/" + folder_name + "/encode_map_ave.npy", encode_map) 


data_folder = "patch3D_40_zabs"
pth = "/results/" + data_folder +"/"+ folder_name 
print(pth)
allmse=[]
allpsnr=[]
allssim =[]
for f in range(0,5):
    df = get_test(data_folder, f)
    model_pth = pth + "/model_f" + str(f) + ".pth"
    mse, psnr,ssim = get_mse(df, model_pth)
    print(mse, psnr,ssim)
    allmse.append(mse)
    allpsnr.append(psnr)
    allssim.append(ssim)
print("")
print(np.round(np.mean(allmse),3),np.round(np.mean(allpsnr),3),np.round(np.mean(allssim),3))
print(np.round(np.std(allmse),3),np.round(np.std(allpsnr),3),np.round(np.std(allssim),3))    
