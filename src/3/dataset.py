import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='Husformer', split_type='train'):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'.pkl'  )
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.m1 = torch.tensor(dataset[split_type]['modality_1'].astype(np.float32)).cpu().detach()
        self.m2 = torch.tensor(dataset[split_type]['modality_2'].astype(np.float32)).cpu().detach()
        self.m3 = torch.tensor(dataset[split_type]['modality_3'].astype(np.float32)).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['label'].astype(np.float32)).cpu().detach()
        self.meta = dataset[split_type]['id']

        self.data = data
        
        self.n_modalities = 3
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.m1.shape[1], self.m2.shape[1], self.m3.shape[1]
    def get_dim(self):
        return self.m1.shape[2], self.m2.shape[2], self.m3.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.m1[index], self.m2[index], self.m3[index])
        Y = self.labels[index]
        META = self.meta[index][0] 
        return X, Y, META        

