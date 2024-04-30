import numpy as np
from tqdm import tqdm
import torch
import wandb
import pdb

class Trainer():
    def __init__(self, model, criterion_rec, dataloader_train, dataloader_val, optimizer, device='cuda', path2save='checkpoint') -> None:
        self.model = model
        self.criterion_rec = criterion_rec
        # self.criterion_dis = criterion_dis
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.device='cuda'
        self.optimizer = optimizer
        self.path2save = path2save
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,80], gamma=0.1)

    def run_trainer(self, n_epochs):
        for epoch in tqdm(range(n_epochs)):
            #train
            loss_rec_train = self.train()
            loss_rec_val = self.val()
            print('Train: Recon - {}\n'.format(loss_rec_train))
            print('Val: Recon - {}\n'.format(loss_rec_val))

            # Log the loss and accuracy values at the end of each epoch
            # print(self.scheduler.get_last_lr()[0])
            lr = float(self.scheduler.get_last_lr()[0])
            # pdb.set_trace()
            wandb.log({
                "Epoch": epoch,
                "Train Loss Recon": loss_rec_train,
                "Val Loss Recon": loss_rec_val,
                "Learning rate": lr})

            # if epoch > 50 and np.mean(total_loss_train) < 1e-3:
            if epoch > 30:
                ## save
                # torch.save(self.model, self.path2save.format(epoch+1)) 
                torch.save(self.model, self.path2save) 
                # break



    def train(self):
        
        loss_rec_epoch = []
 

        self.model.train()
        for batch_x, batch_y in tqdm(self.dataloader_train):
            self.optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            y, z = self.model(batch_x)
            
            # Measure loss
            loss_rec_batch = self.criterion_rec(batch_y, y)

            # Optimize
            loss_rec_batch.backward()
            self.optimizer.step()
            
            loss_rec_epoch.append(loss_rec_batch.item()) 
            
        self.scheduler.step()
        
        return np.mean(loss_rec_epoch)

    def val(self):
        loss_rec_epoch = []


        self.model.eval()
        for batch_x, batch_y in tqdm(self.dataloader_val):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            y, z= self.model(batch_x)
            
            # Measure loss
            loss_rec_batch = self.criterion_rec(batch_y, y)
  
            
            loss_rec_epoch.append(loss_rec_batch.item()) 

        
        return np.mean(loss_rec_epoch)