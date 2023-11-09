import numpy as np
import nibabel as nib
import pandas as pd
import os


lpatch = 40

p_sub = 50
for i in range(450,502):
    p = 0
    while p < p_sub:
        aa = np.nan_to_num(nib.load("All_maps/LRP/zabs_map"+str(i) + ".nii.gz" ).get_fdata())
        bb = np.nan_to_num(nib.load("All_maps/IG/zabs_map"+str(i)+ ".nii.gz").get_fdata())
        cc = np.nan_to_num(nib.load("All_maps/GGC/zabs_map"+str(i)+ ".nii.gz").get_fdata())

        locX = np.random.randint(0,193 - lpatch)
        locY = np.random.randint(0,229 - lpatch)
        locZ = np.random.randint(0,193 - lpatch)

        allmaps = np.concatenate([
            np.expand_dims(aa[locX:locX + lpatch, locY: locY + lpatch,locZ: locZ + lpatch],axis =0),
            np.expand_dims(bb[locX:locX + lpatch, locY: locY + lpatch,locZ: locZ + lpatch],axis =0),
            np.expand_dims(cc[locX:locX + lpatch, locY: locY + lpatch,locZ: locZ + lpatch],axis =0)],axis=0)

        name = "sub" + str(i) + "_" + str(locX) + "_" + str(locY)+ "_" + str(locZ) + ".npy"
        
        if sum(allmaps.flatten()) > 0:
            p += 1      
            print(name, allmaps.shape, sum(allmaps.flatten()))
            np.save("All_maps/patch3D_40_z/" + name, allmaps)
    