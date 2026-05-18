"""Custom dataset and datamodule for folder-based anomaly detection.

Wraps a simple directory layout (train/good, test/good, test/anomaly) into
the anomalib Dataset/DataModule API so any anomalib model can be trained on
this dataset without changes to the model code.
"""
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
    """Dataset for a folder-based anomaly detection layout.

    Expected directory structure:
        root/
          train/good/      ← normal images used for training
          test/good/       ← normal images used for evaluation
          test/anomaly/    ← anomalous images used for evaluation

    No masks are required since this is image-level classification only.
    """

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
        """Build a DataFrame with one row per image.

        anomalib's base class expects a DataFrame with at least
        'image_path', 'split', and 'label_index' columns.
        mask_path is None because we only do image-level classification,
        not pixel-level segmentation.
        """
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
        # num_workers=0 avoids issues on Windows
        num_workers: int = 0,
        name: str = "Datamodule",
    ):
        self.root = root
        self._name = name
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            # Mirror val and test splits so validation sees the same images as
            # the final test, maximising the use of the limited labelled data
            val_split_mode=ValSplitMode.SAME_AS_TEST,
            val_split_ratio=0.5,
            # FROM_DIR reads the test set directly from test/ subfolders
            # instead of splitting it off from the training data
            test_split_mode=TestSplitMode.FROM_DIR,
            test_split_ratio=0.2,
            seed=0,
        )

    @property
    def name(self) -> str:
        return self._name

    def _setup(self, _stage: str | None = None) -> None:
        # _stage is a PyTorch Lightning convention ("fit", "test", etc.);
        # we ignore it and always build both splits at once.
        self.train_data = ImageDataset(root=self.root, split="train")
        self.test_data  = ImageDataset(root=self.root, split="test")


# Backward-compat aliases for checkpoints saved when these classes had other names.
robotv3Datamodule = Datamodule
robotV3Datamodule = Datamodule
CustomDataModule = Datamodule
CustomDataset = ImageDataset
robotv3Dataset = ImageDataset
