import torch
from torchvision.transforms import RandomHorizontalFlip, RandomCrop


class AugMult:
    def __init__(self, crop_size: int = 32, padding: int = 4, k: int = 4):
        """
        Args:
            k: Number of augmentation multiplicities
        """
        if type(k) is not int or k < 2:
            self.k = 0
        else:
            self.k = k - 1
        self.random_flip = RandomHorizontalFlip()
        self.random_crop = RandomCrop(crop_size, padding, padding_mode="reflect")

    def de_et_al(self, x):
        "Augmentation multiplicity according to De et al. (2022)."
        return torch.stack([x] + [self.random_flip(self.random_crop(x))] * self.k)
