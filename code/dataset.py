import os
import numpy as np
from torch.utils.data.dataset import Dataset
import nibabel as ni
import torch
from scipy.ndimage import zoom
import torchio as tio
import random

class MRI_dataloader(Dataset):
    def __init__(self, path, filenames, train,img_size=(137, 128, 128)):
        self.path = path
        # self.filenames = [i for i in os.listdir(path) if i.endswith(".nii")]
        self.filenames = filenames
        #self.upsample = torch.nn.Upsample(size=(137, 128, 128), mode='trilinear', align_corners=True)
        self.img_size = img_size
        transforms_list = [tio.RandomBlur(), tio.RandomBiasField(),
                  tio.RandomSpike(), tio.RandomGhosting(), tio.RandomMotion(),
                  tio.RandomGamma(), tio.RandomNoise()]
        self.transform = tio.OneOf(transforms_list)
        self.train = train
        
    def __getitem__(self, index):
        image = ni.load(os.path.join(self.path, self.filenames[index]))
        image = np.array(image.dataobj)
        image = np.moveaxis(image, [0, 1], [1, 0])
        img_scale = tuple([self.img_size[i]/image.shape[i] for i in range(len(image.shape))])
        image = zoom(image, img_scale)
        #image = image / 255.
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        #image = torch.permute(image, (1, 0, 2)).unsqueeze(0)
        if self.train:

            if random.random() <= 0.8:
                image_ = image.permute((0,2,3,1))
                image_ = self.transform(image_)
                image_ = image_.permute((0,3,1,2))
                #print('in transform - ', image_.shape, image.shape)
                return image_, image 
            else:
                #print('Normal - ', image.shape)
                return image, image
        
        return image, image
    
    def __len__(self):
        return len(self.filenames)