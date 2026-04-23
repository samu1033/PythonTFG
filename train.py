import os
import mlflow
import mlflow.pytorch

from anomalib.engine import Engine
from anomalib.models import Fre, Padim, Fastflow, EfficientAd
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Resize
from anomalib.metrics import AUROC, F1Score
from anomalib.metrics.evaluator import Evaluator
from anomalib.loggers import AnomalibMLFlowLogger

from dataset import kittingRobotDatamodule

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.pytorch.autolog(log_models=True, log_every_n_epoch=1, silent=True)


def main() -> None:
    datamodule = kittingRobotDatamodule(root="./datasets/kittingRobot", train_batch_size=32, eval_batch_size=32)

    evaluator = Evaluator(
        val_metrics=[
            AUROC(fields=["pred_score", "gt_label"]),
        ],
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"]),
            F1Score(fields=["pred_label", "gt_label"]),
        ],
    )

    mlflow_logger = AnomalibMLFlowLogger(
        experiment_name="Anomaly Detection",
        run_name="Padim Real robot",
        log_model="all",
        save_dir="./mlruns",
    )

    model = Padim(
        backbone="resnet18",
        n_features= 100
    )
    

    engine = Engine(
        accelerator="gpu",
        max_epochs=20,
        logger=mlflow_logger,
    )

    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
