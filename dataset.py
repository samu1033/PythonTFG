from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Image as TVImage

from anomalib.data.dataclasses.torch import ImageItem
from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, read_image


class CustomDataset(AnomalibDataset):
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

    @staticmethod
    def get_dataloaders(
        train_ds: "CustomDataset",
        test_ds: "CustomDataset",
        batch_size: int = 32,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, collate_fn=train_ds.collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=test_ds.collate_fn)
        val_loader   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=test_ds.collate_fn)
        return train_loader, test_loader, val_loader


class SingleImageDataset(Dataset):
    def __init__(self, image_path: str | Path):
        self.image_path = str(image_path)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> ImageItem:
        image = TVImage(read_image(self.image_path, as_tensor=True))
        return ImageItem(image=image, image_path=self.image_path)