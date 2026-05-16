from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Image as TVImage

from anomalib.data.dataclasses.torch import ImageItem
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, read_image
from anomalib.data.utils.split import TestSplitMode, ValSplitMode


class ImageDataset(AnomalibDataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        task: str = "classification",
        augmentations: Transform | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)
        self.root = Path(root)
        self.split = split
        self.task_type = task
        self.samples = self._make_samples()

    def _make_samples(self) -> pd.DataFrame:
        samples = []

        if self.split == "train":
            for p in (self.root / "train/good").glob("*.png"):
                samples.append({"image_path": str(p), "split": "train", "label_index": LabelName.NORMAL, "mask_path": None})

        elif self.split == "test":
            for p in (self.root / "test/anomaly").glob("*.png"):
                samples.append({"image_path": str(p), "split": "test", "label_index": LabelName.ABNORMAL, "mask_path": None})
            for p in (self.root / "test/good").glob("*.png"):
                samples.append({"image_path": str(p), "split": "test", "label_index": LabelName.NORMAL, "mask_path": None})

        df = pd.DataFrame(samples)
        df.attrs["task"] = self.task_type
        return df
    
class Datamodule(AnomalibDataModule):
    """AnomalibDataModule for the custom robot dataset."""

    def __init__(
        self,
        root: str | Path = "./datasets/grippy",
        train_batch_size: int = 1,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        name: str = "Datamodule",
    ):
        self.root = root
        self._name = name
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=ValSplitMode.SAME_AS_TEST,
            val_split_ratio=0.5,
            test_split_mode=TestSplitMode.FROM_DIR,
            test_split_ratio=0.2,
            seed=0,
        )

    @property
    def name(self) -> str:
        return self._name

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = ImageDataset(root=self.root, split="train")
        self.test_data  = ImageDataset(root=self.root, split="test")
    


# Backward-compat aliases for checkpoints saved when these classes had other names.
robotv3Datamodule = Datamodule
robotV3Datamodule = Datamodule
CustomDataModule = Datamodule
CustomDataset = ImageDataset
robotv3Dataset = ImageDataset


class SingleImageDataset(Dataset):
    def __init__(self, image_path: str | Path):
        self.image_path = str(image_path)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> ImageItem:
        image = TVImage(read_image(self.image_path, as_tensor=True))
        return ImageItem(image=image, image_path=self.image_path)   