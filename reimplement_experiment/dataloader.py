import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from scipy.io import loadmat
import pdb

class PaperDataset(Dataset):

    def __init__(self, path_to_feature, fold):
        m = loadmat(path_to_feature)
        if fold=='train':
            #pdb.set_trace()
            self.x = np.float32(m['train_features'])
            self.y = m['train_labels'][1][0] # 1 is index of car
        elif fold=='test':
            self.x = np.float32(m['test_features'])
            self.y = m['test_labels'][1][0]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class AllDataset(Dataset):

    def __init__(self, path_to_feature_folder, fold):
        m1 = loadmat(path_to_feature_folder + 'Caltech101.mat')
        m2 = loadmat(path_to_feature_folder + 'LabelMeSpain.mat')
        m3 = loadmat(path_to_feature_folder + 'PASCAL2007.mat')
        m4 = loadmat(path_to_feature_folder + 'SUN09.mat')
        m = [m1, m2, m3, m4]
        x, y, dataset_id = [], [], []
        if fold=='train':
            for i, dataset in enumerate(m):
                x.append(np.float32(dataset['train_features']))
                y.append(dataset['train_labels'][1][0])
                dataset_id.append(np.ones(len(dataset['train_features'])) * i)
            self.x = np.concatenate(x, axis=0)
            self.y = np.concatenate(y, axis=0)
            self.dataset_id = np.concatenate(dataset_id, axis=0)
        else:
            self.x = np.float32(m4['test_features'])
            self.y = m4['test_labels'][1][0]
            self.dataset_id = np.ones(len(m4['test_features'])) * 3

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.dataset_id[idx]


class ExternalDataset(Dataset):

    def __init__(self, path_to_feature_folder, fold, holdname):
        m1 = loadmat(path_to_feature_folder + 'Caltech101.mat')
        m2 = loadmat(path_to_feature_folder + 'LabelMeSpain.mat')
        m3 = loadmat(path_to_feature_folder + 'PASCAL2007.mat')
        m4 = loadmat(path_to_feature_folder + 'SUN09.mat')
        #m = [m1, m2, m3, m4]
        x, y, dataset_id = [], [], []
        if fold=='train':
            if holdname=='Caltech101':
                m = [m2, m3, m4]
            elif holdname=='LabelMeSpain':
                m = [m1, m3, m4]
            elif holdname=='PASCAL2007':
                m = [m1, m2, m4]
            elif holdname=='SUN09':
                m = [m1, m2, m3]
            else:
                assert False, 'dataset name error!'

            for i, dataset in enumerate(m):
                x.append(np.float32(dataset['train_features']))
                y.append(dataset['train_labels'][1][0])
                dataset_id.append(np.ones(len(dataset['train_features'])) * i)
            self.x = np.concatenate(x, axis=0)
            self.y = np.concatenate(y, axis=0)
            self.dataset_id = np.concatenate(dataset_id, axis=0)
        else:
            if holdname=='Caltech101':
                m = m1
            elif holdname=='LabelMeSpain':
                m = m2
            elif holdname=='PASCAL2007':
                m = m3
            elif holdname=='SUN09':
                m = m4
            else:
                assert False, 'dataset name error!'
            self.x = np.float32(m['test_features'])
            self.y = m['test_labels'][1][0]
            self.dataset_id = np.ones(len(m['test_features'])) # it doesn't matter which id we use, 
                                                               #we are going to use visual world weight anyway

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.dataset_id[idx]




if __name__ == '__main__':
    data = AllDataset('./feature/', 'train')
