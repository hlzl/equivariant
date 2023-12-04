import hydra
import torch
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)
from torch.utils.data import DataLoader
from opacus.data_loader import DPDataLoader

from .imagenet import TinyImageNet, ImageNette
from .augmentation import AugMult


class CustomDataloader:
    def __init__(self, cfg):
        """Load dataset for given path with predefined transforms.
        Supported datasets:
        - CIFAR-10
        - CIFAR-100
        - MedMNIST
        - TinyImageNet
        - ImageNette
        - ImageNet32
        - ImageNet
        - RadImageNet
        """
        self.num_workers = cfg.hardware.num_workers
        # Set augmentation multiplicity to 1 (no augmentations) if not specified
        self.aug_mult = int(cfg.setup.params.get("aug_mult") or 1)
        Augmentation = AugMult(
            k=self.aug_mult, crop_size=cfg.setup.dataset.image_size
        ).de_et_al

        # CIFAR-10 or CIFAR-100
        if "CIFAR" in cfg.setup.dataset.name:
            self.train_ds = hydra.utils.instantiate(
                cfg.setup.dataset.func,
                train=True,
                download=True,
                transform=Compose(
                    [
                        ToTensor(),
                        Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                        Augmentation,
                    ],
                ),
            )
            self.val_ds = hydra.utils.instantiate(
                cfg.setup.dataset.func,
                train=(False if cfg.setup.dataset.val_subset == 0.0 else True),
                transform=Compose(
                    [
                        ToTensor(),
                        Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                    ]
                ),
            )
            split_train_set = False if cfg.setup.dataset.val_subset == 0.0 else True

            self.test_ds = hydra.utils.instantiate(
                cfg.setup.dataset.func,
                train=False,
                download=True,
                transform=Compose(
                    [
                        ToTensor(),
                        Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                    ],
                ),
            )
            self.n_channels = 3
            self.n_classes = int(cfg.setup.dataset.name.split("CIFAR")[-1])

        # TinyImageNet
        elif cfg.setup.dataset.name == "tinyimagenet":
            # Tiny: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
            try:
                self.train_ds = TinyImageNet(
                    cfg.setup.dataset.path,
                    mode="train",
                    preload=True,
                    load_transform=None,
                    transform=Compose(
                        [
                            ToTensor(),
                            Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                            Augmentation,
                        ]
                    ),
                )
                # NOTE: Set validation set to test set is subset size is set to 0.0
                self.val_ds = TinyImageNet(
                    cfg.setup.dataset.path,
                    mode=("val" if cfg.setup.dataset.val_subset == 0.0 else "train"),
                    preload=True,
                    load_transform=None,
                    transform=Compose(
                        [
                            ToTensor(),
                            Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
                split_train_set = False if cfg.setup.dataset.val_subset == 0.0 else True

                self.test_ds = TinyImageNet(
                    cfg.setup.dataset.path,
                    mode="val",
                    preload=True,
                    load_transform=None,
                    transform=Compose(
                        [
                            ToTensor(),
                            Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
            except ValueError:
                raise ValueError(
                    f"The following directory has no suitable dataset:\n{cfg.setup.dataset.path}"
                )

            self.n_channels = 3
            self.n_classes = 200

        # ImageNette
        elif cfg.setup.dataset.name == "imagenette":
            # ImageNette: https://github.com/fastai/imagenette
            try:
                self.train_ds = ImageNette(
                    cfg.setup.dataset.path,
                    mode="train",
                    preload=True,
                    load_transform=None,
                    transform=Compose(
                        [
                            ToTensor(),
                            Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                            Augmentation,
                        ]
                    ),
                )
                self.val_ds = ImageNette(
                    cfg.setup.dataset.path,
                    mode=("val" if cfg.setup.dataset.val_subset == 0.0 else "train"),
                    preload=True,
                    load_transform=None,
                    transform=Compose(
                        [
                            ToTensor(),
                            Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
                split_train_set = False if cfg.setup.dataset.val_subset == 0.0 else True

                self.test_ds = ImageNette(
                    cfg.setup.dataset.path,
                    mode="val",
                    preload=True,
                    load_transform=None,
                    transform=Compose(
                        [
                            ToTensor(),
                            Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
            except ValueError:
                raise ValueError(
                    f"The following directory has no suitable dataset:\n{cfg.setup.dataset.path}"
                )

            self.n_channels = 3
            self.n_classes = 10

        else:
            raise ValueError(
                f"Dataset {cfg.setup.dataset.name} currently not supported."
            )

        if split_train_set:
            self.split_train_set(cfg)

    def split_train_set(self, cfg):
        """Check if train and val samples overlap if both are subsets of the train set."""
        if (
            float(cfg.setup.dataset.get("train_subset") or 0.0)
            + float(cfg.setup.dataset.get("val_subset") or 0.0)
        ) <= 1.0:
            indices = torch.randperm(len(self.train_ds))

            # Restrict train set to percentage of samples stated in config
            train_size = int(
                float(cfg.setup.dataset.get("train_subset") or 0.0) * len(self.train_ds)
            )
            if train_size > 0:
                self.train_ds = torch.utils.data.Subset(
                    self.train_ds, indices[:train_size]
                )
            else:
                self.train_ds = None

            # Check if test set already exists - otherwise use subset of train samples
            if (
                self.test_ds is None
                and cfg.setup.dataset.get("test_subset") is not None
            ):
                test_size = -int(
                    cfg.setup.dataset.test_subset * len(self.val_ds)
                )  # negative to make it easier useable for val subset
                self.test_ds = torch.utils.data.Subset(self.val_ds, indices[test_size:])
            else:
                test_size = None

            # Create validation set as subset of training samples if percentage larger than 0
            val_size = int(
                float(cfg.setup.dataset.get("val_subset") or 0.0) * len(self.val_ds)
            )
            if val_size > 0:
                # Additionally substract test subset if applicable - otherwise test_size is None and nothing happens
                self.val_ds = torch.utils.data.Subset(
                    self.val_ds,
                    indices[-(val_size - int(test_size or 0)) : test_size],
                )
            else:
                self.val_ds = None
        else:
            raise ValueError(
                f"Train and validation set overlap! If your validation samples are a subset"
                f" of your train set, make sure you set train_subset and val_subset correctly "
                f"(both combined should be <= 1.0 but are {cfg.setup.dataset.train_subset}"
                f" and {cfg.setup.dataset.val_subset} respectively)."
            )

    def get_dataloader(self, train_bs: int, test_bs: int, cpu_generator=None):
        """Get dataloader from previously created datasets for a certain compute setup."""
        train_loader = DataLoader(
            self.train_ds,
            batch_size=train_bs,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=10,
            drop_last=True,
        )
        if train_bs < 8:  # small BS can cause problems with random sampling
            print(
                "Batch size is smaller than 8 - might cause problems with random sampling."
            )
        train_loader = DPDataLoader.from_data_loader(
            train_loader, generator=cpu_generator, distributed=False
        )

        if self.val_ds is not None:
            val_loader = DataLoader(
                self.val_ds,
                batch_size=test_bs,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=10,
            )
        else:
            val_loader = None

        test_loader = DataLoader(
            self.test_ds,
            batch_size=test_bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=10,
        )

        return train_loader, val_loader, test_loader
