import torch
from torchvision import utils as vutils
import os


def save_image(tensor, filepath, nrow=1, normalize=True, scale_each=True):
    """
    Save a PyTorch tensor as an image file.

    Args:
        tensor (torch.Tensor or list): Tensor to save to an image. If the input is a list, it is stacked into a tensor.
        filepath (str): Where to save the image
        nrow (int, optional): Number of images per row. Defaults to 1.
        normalize (bool, optional): Whether to normalize values from [-1,1] to [0,1] through clamping. Defaults to True.
        scale_each (bool, optional): Whether to scale each image individually, defaults to True.

    Returns:
        bool: Whether the image was saved
    """
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)

    tensor = tensor.cpu()

    if normalize:
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        vutils.save_image(
            tensor, filepath, nrow=nrow, normalize=False, scale_each=scale_each
        )
        return True
    except Exception as e:
        print(f"Error saving image to {filepath}: {e}")
        return False


def save_comparison_grid(real_A, fake_B, real_B, fake_A, filepath, normalize=True):
    """
    Save a grid of real and generated images for comparison.

    Args:
        real_A (torch.Tensor): Original images from domain A
            Shape: (N,C,H,W) where N is batch size
        fake_B (torch.Tensor): Generated images in domain B
            Shape: (N,C,H,W)
        real_B (torch.Tensor): Original images from domain B
            Shape: (N,C,H,W)
        fake_A (torch.Tensor): Generated images in domain A
            Shape: (N,C,H,W)
        filepath (str): Path where the comparison grid will be saved
        normalize (bool, optional): Whether to normalize the images.
            Defaults to True.

    Returns:
        bool: True if save successful, False otherwise
    """
    n = min(real_A.size(0), 4)

    real_A = real_A[:n]
    fake_B = fake_B[:n]
    real_B = real_B[:n]
    fake_A = fake_A[:n]

    top_row = torch.cat([real_A, fake_B], dim=3)
    bottom_row = torch.cat([real_B, fake_A], dim=3)
    grid = torch.cat([top_row, bottom_row], dim=2)

    return save_image(grid, filepath, nrow=1, normalize=normalize)
