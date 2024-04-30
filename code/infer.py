import torch
import numpy as np
import nibabel as ni
import os, shutil
import time
import random
import pandas as pd 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import loss
from dataset import MRI_dataloader
from model import VAE
from tqdm import tqdm

device = 'cuda'

ckpt_path = path2save = "./checkpoint/model_vae_epoch_{}.pt"
ckpt_path = ckpt_path.format(1)
path = 'dataset/images'
mri_loader = MRI_dataloader(path)
batch_size = 8
dataloader = DataLoader(mri_loader, batch_size=batch_size, shuffle=True)
model = VAE().to(device)

model.eval()
for batch in tqdm(dataloader):
    batch = batch.to(device)
    y, z_mean, z_log_sigma = model(batch)
    print(y.shape)

    break