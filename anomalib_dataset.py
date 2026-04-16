from pathlib import Path

import pandas as pd
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName


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