import numpy as np
import cv2 
import pandas as pd
import os
import torch
import pickle
from torch.utils.data import Dataset

def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_image_and_landmarks(data_path, image_rel_path, src_landmarks_in_pickle=True):
    idx, img_name = image_rel_path.split('/')
    if src_landmarks_in_pickle:
        landmarks_name = img_name.split('.')[0] + '.landmark'
        landmarks = load(os.path.join(data_path, idx, landmarks_name))
    else:
        df = pd.read_csv(os.path.join(data_path, idx, 'info.csv')).rename(columns={'Unnamed: 0':'img_id'})
        landmarks_raw = df[df['img_id'] == img_name]['4'].iloc[0]
        landmarks = parse_keypoints(landmarks_raw)
    
    img = imread(os.path.join(data_path, image_rel_path))
    return img, landmarks

def imread(path):
    img = cv2.imread(os.path.join(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def parse_keypoints(kp_raw):
    data = list(kp_raw.apply(eval))
    return np.array(data)
    
class TwinPairsDataset(Dataset):
    def __init__(self, dataroot, df_pairs, df_views, transform, keypoints=False, id_columns=['id_1', 'id_2'], landmarks=False):
        '''
        dataroot: path to folder with items
        df_pairs: pd dataframe containing pairs of ids and a correspodind label:
                    'Same', 'Fraternal', 'Identical', 'UnknownTwinType',
                    'IdenticalMirror', 'Sibling', 'IdenticalTriplet'
        df_views: pd dataframe containing list of available for each id in the dataset                    
        transform: torchvision transform
        '''
        self.dataroot = dataroot
        self.df_pairs = df_pairs
        self.df_views = df_views
        self.transform = transform
        self.keypoints = keypoints
        self.id_columns = id_columns
        self.n = len(id_columns)
        self.landmarks = landmarks
        
    def __getitem__(self, index):
        def get_img_path(person_id, view):
            path = os.path.join(self.dataroot, person_id, view)
            return imread(path)
        
        ids = self.df_pairs.iloc[index][self.id_columns].values
        ids = [str(x) for x in ids]
    
        label = self.df_pairs.iloc[index].label=='Same'
        
        if ids[0]==ids[1]:
             views = np.random.choice(self.df_views.loc[ids[0]]['filename'], size=2, replace=False) 
        else:
            views = [np.random.choice(self.df_views.loc[ids[i]]['filename']) for i in range(self.n)]

        images=[]
        keypoints=[]
#         paths = [os.path.join(self.dataroot, ids[i], views[i]) for i in range(self.n)]
        for i in range(self.n):
            subject_id = ids[i]
            view=views[i]
            img, kp = get_image_and_landmarks(self.dataroot, f'{subject_id}/{view}')
            images.append(img)# = [imread(path) for path in paths]
            keypoints.append(kp)
        
        landmark_images = []
        if self.landmarks:
            landmark_images = [imread(os.path.join(self.dataroot, ids[i], views[i].split('.')[0]+'_landmark.jpg')) for i in range(self.n)]
        
       
        sample = dict([(f'image{i}',images[i]) for i in range(self.n)])
        
        if self.keypoints:
            kp = [pd.read_csv(os.path.join(self.dataroot, ids[i], 'keypoints.csv')) for i in range(self.n)]
            keypoints = [parse_keypoints(kp[i][views[i]]) for i in range(self.n)]
            
            sample.update(dict([(f'keypoints{i}',keypoints[i]) for i in range(self.n)]))
        images_l = []
#         for i in range(self.n):
#             images_l.append(np.stack([images[i], landmark_images[i]])
        if self.transform:
            samples = [{'image':image } for image in images]
            
            if self.keypoints:
                for i in range(self.n):
                    samples[i]['keypoints'] = keypoints[i]
            if self.landmarks:
                for i in range(self.n):
                    samples[i]['image1'] = landmark_images[i]
                
            augs = [self.transform(**sample) for sample in samples]            
            
            if self.landmarks:
                sample = dict([(f'image{i}', torch.cat([augs[i]['image'], augs[i]['image1']], 0)) for i in range(self.n)])
            else:
                sample = dict([(f'image{i}', augs[i]['image']) for i in range(self.n)])

            if self.keypoints:
                sample.update(dict([(f'keypoints{i}',np.array(augs[i]['keypoints'])) for i in range(self.n)]))
        
        sample['label'] = label   
        return {'img1': sample['image0'], 'img2': sample['image1'], 'is_same': label, 'id0': ids[0], 'id1':[1]}
    
    def __len__(self):
        return self.df_pairs.shape[0]
    
    
class ClassificationDataset(Dataset):
    def __init__(self, dataroot, df_views, transform, keypoints=False, landmarks=False):
        '''
        dataroot: path to folder with items
        df_views: pd dataframe containing list of available for each id in the dataset                    
        transform: torchvision transform
        '''
        self.dataroot = dataroot
        self.df_views = df_views
        self.transform = transform
        self.keypoints = keypoints
        self.landmarks = landmarks
        
    def __getitem__(self, index):
        def get_img_path(person_id, view):
            path = os.path.join(self.dataroot, person_id, view)
            return imread(path)
        index = index%self.df_views.shape[0]
        views, subject_id, label  = self.df_views.iloc[index].values
        subject_id = str(subject_id)
        view = np.random.choice(views) 

        path = os.path.join(self.dataroot, subject_id, view)
        
        img, kp = get_image_and_landmarks(self.dataroot, f'{subject_id}/{view}')    
            
        sample = {'image': img, 'label': label}
        
        if self.landmarks:
            sample['image1'] = imread(os.path.join(self.dataroot, subject_id, view.split('.')[0]+'_landmark.jpg'))
             
        
        if self.keypoints:
#             kp = pd.read_csv(os.path.join(self.dataroot, subject_id, 'keypoints.csv'))
            sample['keypoints'] = parse_keypoints(kp)
        
        if self.transform:            
            sample = self.transform(**sample)
            if self.keypoints:
                sample['keypoints'] = np.array(sample['keypoints'])
            if self.landmarks:
                sample['image'] = torch.cat([sample['image'], sample['image1']], 0)
           
        return {'img': sample['image'], 'label': label, 'instance': view}
    
    def get_num_classes(self):
        return self.df_views.shape[0]
    
    def __len__(self):
        return self.df_views.shape[0]*6
    