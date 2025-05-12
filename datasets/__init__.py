from torch.utils.data import Dataset
from torch import Tensor


class ImageDataset(Dataset):
    """Dataset of RGB images."""
    def __init__(self, data: Tensor):
        self.data = data # Shape: (50000, 3, 32, 32) (example)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
