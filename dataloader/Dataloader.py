import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from PIL import Image

import torch
import torch.nn as nn

import torchvision
from torch.utils.data import DataLoader, Dataset

% matplotlib inline
warnings.filterwarnings('ignore')


class TwinsDataloader(Dataset):
    def __init__(self, dataroot, df, transform):
        '''
        dataroot: path to folder with items
        df: pandas dataframe with fields view, id_a, id_b
        transform: torchvision transform
        '''
        self.dataroot = dataroot
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
        
        def get_img_path(img_id, view):
            #return os.path.join(self.dataroot, f'{img_id}/{img_id}d{view}__face.jpg')
            return self.dataroot+f'{img_id}/{img_id}d{view}__face.jpg'
        
        #print(self.df.iloc[index].values[0])
        
        view, id_a, id_b  = self.df.iloc[index].values
        #print(view)
        #view = np.random.choice(views)
        #print(view, id_a, id_b)
        
        path_a = get_img_path(id_a, view)
        path_b = get_img_path(id_b, view)

        img_a = Image.open(path_a)
        img_b = Image.open(path_b)

        #plt.imshow(img_a)
        #plt.show()
        #plt.imshow(img_b)
        
        img_a = self.transform(img_a)
        img_b = self.transform(img_b)
        
        return {'img_a': img_a, 'img_b': img_b, 'class_a':id_a,'class_b':id_b}#'A_paths': path_a, 'B_paths': path_b }

    def __len__(self):
        return self.df.shape[0]