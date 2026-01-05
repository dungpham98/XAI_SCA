import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

def print_structure(name, obj):
    print(name)

class SCADataset(Dataset):
    def __init__(self, file_path, split='train', input_length=None):
        self.file_path = file_path
        self.split = split
        
        # Open h5 file
        self.h5_file = h5py.File(file_path, 'r')
        '''
        with h5py.File(file_path, 'r') as f:
            print(f"Keys in root: {list(f.keys())}")
            # This will recursively print every group and dataset inside
            f.visititems(print_structure)
        '''
        if  'AES' in file_path:
            if split == 'train':
                self.group = self.h5_file['D1/Unprotected/Profiling']
            elif split == 'test' or split == 'attack':
                self.group = self.h5_file['D1/Unprotected/Attack']
            else:
                raise ValueError("Split must be 'train' or 'test'")
            
            self.traces = self.group['Traces']
            self.labels = self.group['Labels']
            # Metadata often needed for attacks
            self.metadata = self.group['MetaData'] 
            
            self.input_length = input_length if input_length else self.traces.shape[1]
            print(self.input_length)
        else:
            # Select correct group based on ASCAD structure
            if split == 'train':
                self.group = self.h5_file['Profiling_traces']
            elif split == 'test' or split == 'attack':
                self.group = self.h5_file['Attack_traces']
            else:
                raise ValueError("Split must be 'train' or 'test'")
                
            self.traces = self.group['traces']
            self.labels = self.group['labels']
            # Metadata often needed for attacks
            self.metadata = self.group['metadata'] 
            
            self.input_length = input_length if input_length else self.traces.shape[1]

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        # Load data on the fly to save RAM
        trace = torch.from_numpy(self.traces[idx, :self.input_length]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Often needed for Key Rank (key, plaintext, etc.)
        # Assuming metadata has fields 'key' and 'plaintext' or similar
        # Adjust field names based on your specific H5 structure
        key = torch.tensor(self.metadata[idx]['key'], dtype=torch.long)
        plaintext = torch.tensor(self.metadata[idx]['plaintext'], dtype=torch.long)
        
        # Add channel dimension for CNN [C, L]
        return trace.unsqueeze(0), label, plaintext, key

def get_dataloaders(file_path, batch_size=256, input_length=700):
    train_dataset = SCADataset(file_path, split='train', input_length=input_length)
    test_dataset = SCADataset(file_path, split='test', input_length=input_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader