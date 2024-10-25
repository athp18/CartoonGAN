from PIL import Image
import os
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """Custom dataset for loading images from directories."""
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        self.files_A = sorted(os.listdir(os.path.join(root, f'{mode}A')))
        self.files_B = sorted(os.listdir(os.path.join(root, f'{mode}B')))
        self.root = root
        self.mode = mode

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        img_A_path = os.path.join(self.root, f'{self.mode}A', self.files_A[idx % len(self.files_A)])
        img_B_path = os.path.join(self.root, f'{self.mode}B', self.files_B[idx % len(self.files_B)])

        img_A = Image.open(img_A_path).convert('RGB')
        img_B = Image.open(img_B_path).convert('RGB')

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}
