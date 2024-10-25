from PIL import Image
import os
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Custom dataset class for loading and processing unpaired image sets for CycleGAN training.

    This dataset is designed for the CycleGAN architecture, which performs unpaired
    image-to-image translation between two domains (A and B). It expects a directory
    structure where images from each domain are stored in separate folders.

    The directory structure should be like this:
        root/
        |-- trainA/ # Images from domain A for training (real images)
        |-- trainB/ # Images from domain B for training (cartoon images)
        |-- testA/ # Images from domain A for testing (real images)
        └── testB/ # Images from domain B for testing (cartoon images)

    Args:
        root (str): Root directory containing the image folders
        transform (callable, optional): Optional transform to be applied to the images. Defaults to None.
        mode (str, optional): Dataset mode, either 'train' or 'test'. Determines which folder to use.

    Returns:
        image_dict (dict): A dictionary containing:
            - 'A': Transformed image from domain A (PIL Image or tensor if transform is applied)
            - 'B': Transformed image from domain B (PIL Image or tensor if transform is applied)

    """

    def __init__(self, root, transform=None, mode="train"):
        self.transform = transform
        self.files_A = sorted(os.listdir(os.path.join(root, f"{mode}A")))
        self.files_B = sorted(os.listdir(os.path.join(root, f"{mode}B")))
        self.root = root
        self.mode = mode

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        img_A_path = os.path.join(
            self.root, f"{self.mode}A", self.files_A[idx % len(self.files_A)]
        )
        img_B_path = os.path.join(
            self.root, f"{self.mode}B", self.files_B[idx % len(self.files_B)]
        )

        img_A = Image.open(img_A_path).convert("RGB")
        img_B = Image.open(img_B_path).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        image_dict = {"A": img_A, "B": img_B}
        return image_dict
