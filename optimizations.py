import torch
import random


class ReplayBuffer:
    """
    Buffer to store previously generated images. This helps avoid Discriminator over-training.

    Args:
        max_size (int, optional): Maximum number of images to store in the buffer.
            Defaults to 50. Higher values provide more variety but use more memory.
    """

    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        """
        Updates buffer with new images and returns a mixture of current and historical images.

        For each image in the input batch:
            1. If buffer is not full: adds image to buffer and returns it
            2. If buffer is full:
               - 50% chance: returns the current image
               - 50% chance: returns a random historical image and replaces it with current

        Args:
            data (torch.Tensor): Batch of newly generated images
                Shape: [B, C, H, W] where B is batch size

        Returns:
            torch.Tensor: Batch of images with same shape as input, containing
                a mix of current and historical generated images
        """
        result = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = element
                    result.append(tmp)
                else:
                    result.append(element)
        return torch.cat(result)
