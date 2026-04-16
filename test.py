

from typing import Any

from anomalib.models.image.fastflow import FastflowModel
from anomalib.models.image.fastflow import Fastflow
from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName
from anomalib.metrics.evaluator import Evaluator

import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self, conv_model: Fastflow):
        super().__init__()
        self.cnn = conv_model #Stored pretrained CNN model
        self.lstm = nn.LSTM(
            input_size= 1,
            num_layers= 1,
            hidden_size= 32,
            batch_first= True
        )
        self.linear = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        cnn_output = self.cnn(x)
        anomalyscore = cnn_output.pred_score.item()
        
ff = Fastflow.load_from_checkpoint("./results/FastFlow/CustomDataModule/v13/weights/lightning/model.ckpt", weights_only=False).model

ff.eval()

with torch.no_grad():
    

