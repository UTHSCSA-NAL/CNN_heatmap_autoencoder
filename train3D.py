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
import pathlib
import os
from utils import *
import time

device = (torch.device('cuda'))
model = ae_max()
model.to(device)

n_epochs = 100
data = "/data_example/patch3D_40_abs/"
lpatch = 20
batch_size = 512
model_name = "ae_max"
out_pth = "/results/" + data + "/" + model_name
pathlib.Path(out_pth).mkdir(parents=True, exist_ok=True)
print(out_pth)


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

l = []
path = "/data_example/patch3D_40_abs/"
for i in os.listdir(path):
    l.append(path +i)
df = pd.DataFrame(l)

train_all = DatasetFromNii(df)
print(len(train_all))
train_loader = torch.utils.data.DataLoader(train_all, batch_size=100, num_workers=0)
print(len(train_loader))


#### Training
t_loss = []
for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    program_starts = time.time()

    for data in train_loader:
        images = data
        images = images.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)

        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss/len(train_loader)
    t_loss.append(train_loss)
    now = time.time()
    print('Epoch: {} \tTime: {:.3f} \tTraining Loss: {:.5f}'.format(epoch, now - program_starts, train_loss))

pd.DataFrame(t_loss).to_csv(out_pth + "/train_loss.csv")
torch.save(model.state_dict(), out_pth + "model.pth")
    


n = 5
l = np.random.randint(0,len(df), n)
fig, axes = plt.subplots(nrows=3, ncols=2*n, figsize = (20,6))

for i in range(0,n):
    a = np.load(df.iloc[l[i]][0])
    model = ConvAutoencoder3D_max()
    model.load_state_dict(torch.load("AE3D.pth"))
    model.eval()
    
    aa = torch.from_numpy(np.expand_dims(a,axis = 0).astype(np.float32))


    out, encode = model(aa)
    out = out[0,:,:,:,:].detach().cpu().numpy()
    
    axes[0][0].set_ylabel('LRP')
    axes[1][0].set_ylabel('IG')
    axes[2][0].set_ylabel('GGC')
    
    axes[0][2*i].imshow(a[0][:,:,int(lpatch/2)])
    axes[1][2*i].imshow(a[1][:,:,int(lpatch/2)])
    axes[2][2*i].imshow(a[2][:,:,int(lpatch/2)])
    axes[0][2*i+1].imshow(out[0][:,:,int(lpatch/2)])
    axes[1][2*i+1].imshow(out[1][:,:,int(lpatch/2)])
    axes[2][2*i+1].imshow(out[2][:,:,int(lpatch/2)])
    
fig.savefig(out_pth + "/visual.jpg")














