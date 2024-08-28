import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

_CLASSNAMES = [
    "A",
    "B",
    "C",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class KaeriDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.imagesize = (3, imagesize, imagesize)

        def __getitem__(self, idx):
            classname, anomaly, image_path = self.data_to_iterate[idx]
            image = PIL.Image.open(image_path).convert("RGB")
            image = self.transform_img(image)

            return {
                "image": image,
                "classname": classname,
                "anomaly": anomaly,
                "is_anomaly": int(anomaly != "good"),
                "image_name": "/".join(image_path.split("/")[-4:]),
                "image_path": image_path,
            }
            
            
        def __len__(self):
            return len(self.data_to_iterate)
        
        def get_image_data(self):
            imgpaths_per_class = {}

            for classname in self.classnames_to_use:
                classpath = os.path.join(self.source, classname, self.split.value)
                anomaly_types = os.listdir(classpath)

                imgpaths_per_class[classname] = {}

                for anomaly in anomaly_types:
                    anomaly_path = os.path.join(classpath, anomaly)
                    anomaly_files = sorted(os.listdir(anomaly_path))
                    imgpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_path, x) for x in anomaly_files
                    ]

                    if self.train_val_split < 1.0:
                        n_images = len(imgpaths_per_class[classname][anomaly])
                        train_val_split_idx = int(n_images * self.train_val_split)
                        if self.split == DatasetSplit.TRAIN:
                            imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                classname
                            ][anomaly][:train_val_split_idx]
                        elif self.split == DatasetSplit.VAL:
                            imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                classname
                            ][anomaly][train_val_split_idx:]

            # Unrolls the data dictionary to an easy-to-iterate list.
            data_to_iterate = []
            for classname in sorted(imgpaths_per_class.keys()):
                for anomaly in sorted(imgpaths_per_class[classname].keys()):
                    for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                        data_tuple = [classname, anomaly, image_path]
                        data_tuple.append(None)
                        data_to_iterate.append(data_tuple)

            return imgpaths_per_class, data_to_iterate