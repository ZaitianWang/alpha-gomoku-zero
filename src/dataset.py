import torch.utils.data as data

class GameDataset(data.Dataset):
    def __init__(self, x):
        self.x = x
    
    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return len(self.x)
    