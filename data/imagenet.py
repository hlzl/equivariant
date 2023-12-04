import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import imageio

__all__ = ["TinyImageNet", "ImageNette"]


def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message.format(URL))


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


class TinyImageNetPaths:
    r"""Creates a paths datastructure for the tiny imagenet.

    TinyImageNetPath
    ├── test
    │   └── images
    │       ├── test_0.JPEG
    │       ├── t...
    │       └── ...
    ├── train
    │   ├── n01443537
    │   │   ├── images
    │   │   │   ├── n01443537_0.JPEG
    │   │   │   ├── n...
    │   │   │   └── ...
    │   │   └── n01443537_boxes.txt
    │   ├── n...
    │   │   ├── images
    │   │   │   ├── ...
    │   │   │   └── ...
    ├── val
    │   ├── images
    │   │   ├── val_0.JPEG
    │   │   ├── v...
    │   │   └── ...
    │   └── val_annotations.txt
    ├── wnids.txt
    └── words.txt

    Args:
        root_dir: Where the data is located
        download: Download if the data is not there

    Attributes:
        label_id:
        ids:
        nit_to_words:
        data_dict:
    """

    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip(
                "http://cs231n.stanford.edu/tiny-imagenet-200.zip", root_dir
            )
        train_path = os.path.join(root_dir, "train")
        val_path = os.path.join(root_dir, "val")
        test_path = os.path.join(root_dir, "test")

        self._make_paths(train_path, val_path, test_path)

    def _make_paths(self, train_path, val_path, test_path):
        self.ids = []
        for nid in os.listdir(train_path):
            nid = nid.strip()
            self.ids.append(nid)

        self.paths = {
            "train": [],  # [img_path, id, nid, box]
            "val": [],  # [img_path, id, nid, box]
            "test": [],  # img_path
        }

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + "_boxes.txt")
            imgs_path = os.path.join(train_path, nid, "images")
            label_id = self.ids.index(nid)
            with open(anno_path, "r") as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths["train"].append((fname, label_id, nid, bbox))

        # Get the validation paths and labels
        with open(os.path.join(val_path, "val_annotations.txt")) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, "images", fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths["val"].append((fname, label_id, nid, bbox))

        # Get the test paths - NOTE: currently not used as there are no labels
        self.paths["test"] = list(
            map(lambda x: os.path.join(test_path, x), os.listdir(test_path))
        )


class TinyImageNet(Dataset):
    r"""Datastructure for the tiny image dataset.
    Args:
        root_dir: Root directory for the data
        mode: One of "train", "test", or "val"
        preload: Preload into memory
        load_transform: Transformation to use at the preload time
        transform: Transformation to use at the retrieval time
        download: Download the dataset

    Attributes:
        tinp: Instance of the TinyImageNetPaths
        img_data: Image data
        label_data: Label data
    """

    def __init__(
        self,
        root_dir,
        mode="train",
        preload=True,
        load_transform=None,
        transform=None,
        download=False,
        max_samples=None,
    ):
        tinp = TinyImageNetPaths(root_dir, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[: self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros(
                (self.samples_num,) + self.IMAGE_SHAPE, dtype=np.float32
            )
            self.label_data = np.zeros((self.samples_num,), dtype=int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.img_data[idx] = img
                if mode != "test":
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            lbl = None if self.mode == "test" else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            lbl = None if self.mode == "test" else s[self.label_idx]
        return self.transform(img), lbl


class ImageNette(Dataset):
    r"""Datastructure for the ImageNette dataset.
    Args:
        root_dir: Root directory for the data
        mode: One of "train" or "val"
        preload: Preload into memory
        load_transform: Transformation to use at the preload time
        transform: Transformation to use at the retrieval time
        download: Download the dataset

    Attributes:
        tinp: Instance of the TinyImageNetPaths
        img_data: Image data
        label_data: Label data
    """

    def __init__(
        self,
        root_dir,
        image_size=160,
        mode="train",
        preload=True,
        load_transform=None,
        transform=None,
        download=False,
        max_samples=None,
    ):
        self.mode = mode
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (image_size, image_size, 3)

        self.img_data = []
        self.label_data = []

        if download:
            download_and_unzip(
                f"https://s3.amazonaws.com/fast-ai-imageclas/imagenette-{image_size}.tgz",
                root_dir,
            )  # add a 2 after /imagenette in the link for version after 2019

        path = os.path.join(root_dir, mode)

        self.ids = []
        for nid in os.listdir(path):
            nid = nid.strip()
            self.ids.append(nid)

        # Get the train and val paths for ImageNette
        self.samples = []
        nids = os.listdir(path)
        for nid in nids:
            imgs_path = os.path.join(path, nid)
            label_id = self.ids.index(nid)
            for file in os.listdir(imgs_path):
                self.samples.append((os.path.join(imgs_path, file), label_id))

        self.samples_num = len(self.samples)

        self.max_samples = max_samples
        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[: self.samples_num]

        if self.preload:
            load_desc = f"Preloading {mode} data..."
            self.img_data = np.zeros(
                (self.samples_num,) + self.IMAGE_SHAPE, dtype=np.float32
            )
            self.label_data = np.zeros((self.samples_num,), dtype=int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.img_data[idx] = img[
                    :image_size, :image_size
                ]  # cropping image - TODO: find better ways
                if mode != "test":
                    self.label_data[idx] = s[1]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            lbl = self.label_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            lbl = s[1]
        return self.transform(img), lbl
