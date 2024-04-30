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
from model_AE import AE
from tqdm import tqdm
from utils import shuffle_and_split
import wandb
from train_helper_AE import Trainer
wandb.login()

random.seed(2024)

wandb.init(name='AE l1 loss', 
           project='AE l1',
           notes='l1 step lr', 
           tags=['MRI data', 'Test Run'])

device = 'cuda'


# path2save = "./checkpoint/model_vae_epoch_{}.pt"
path2save = "./checkpoint/best_model_AE_l1_loss.pt"

path = 'dataset/images'
filenames = [i for i in os.listdir(path) if i.endswith(".nii")]
train_files, val_files = shuffle_and_split(filenames)

with open('./checkpoint/data_stats_AE.txt', 'w') as f:
    
    f.write("----------- Train files ---------\n")
    for line in train_files:
        f.write(f"{line}\n")
    
    f.write("----------- Val files -----------\n")
    for line in val_files:
        f.write(f"{line}\n")
    
    


mri_loader_train = MRI_dataloader(path, filenames=train_files, train=True)
mri_loader_val = MRI_dataloader(path, filenames=val_files, train=False)

batch_size = 8
dataloader_train = DataLoader(mri_loader_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(mri_loader_val, batch_size=batch_size, shuffle=True)

epochs = 50

model = AE().to(device)


wandb.config.lr = 1e-2  
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=5e-7)
# optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.lr, momentum=0.9)

###################
criterion_rec = loss.L1Loss()
# criterion_rec = loss.L2Loss()
# criterion_dis = loss.KLDivergence()


model_trainer = Trainer(model, criterion_rec, dataloader_train, dataloader_val, optimizer, device='cuda', path2save=path2save)
model_trainer.run_trainer(n_epochs=100) 

# model.train()
# for epoch in tqdm(range(epochs)):
#     loss_rec_batch, loss_KL_batch, total_loss_batch = 0, 0, 0
#     loss_rec_epoch, loss_KL_epoch, total_loss_epoch = [], [], []
    
    
    
#     if epoch > 50 and np.mean(total_loss_epoch) < 1e-3:
#         ## save
#         torch.save(model, path2save.format(epoch+1)) 
#         break
                            

#     print(f'Epoch - {epoch+1}, recon loss - {np.mean(loss_rec_epoch)} KL loss - {np.mean(loss_KL_epoch)} Total loss - {np.mean(total_loss_epoch)}')
