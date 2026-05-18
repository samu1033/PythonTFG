"""Training script for anomaly detection models using anomalib.

Supported models: FastFlow, PaDiM, EfficientAD, PatchCore, FRE.
Results and metrics are logged to MLflow under ./mlruns.
Change MODEL and dataset root in the config section to switch experiments.
"""
import os
from anomalib.engine import Engine
from anomalib.models import Fastflow, Padim, EfficientAd, Patchcore, Fre
from anomalib.metrics import AUROC, F1Score
from anomalib.metrics.evaluator import Evaluator
from anomalib.loggers import AnomalibMLFlowLogger
from anomalib.visualization import ImageVisualizer

from dataset import Datamodule

# Enable CPU/GPU and memory usage logging in MLflow alongside model metrics
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

# --- Config ---
MODEL = "padim"  # options: "fastflow", "padim", "efficientad", "patchcore"
MAX_EPOCHS = 10
# ---------------------


def build_model(name: str, evaluator: Evaluator):
    """Factory that instantiates the requested anomaly detection model.

    PaDiM and PatchCore are memory-based (no gradient updates at train time):
    they fit a statistical model over training embeddings.
    FastFlow and EfficientAD involve actual backprop and need more epochs.
    """
    if name == "fastflow":
        return Fastflow(backbone="resnet18", evaluator=evaluator)
    if name == "padim":
        return Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"], n_features=200, pre_trained=True, evaluator=evaluator)
    if name == "efficientad":
        return EfficientAd(imagenet_dir="./datasets/imagenette", model_size="small", evaluator=evaluator)
    if name == "fre":
        return Fre(backbone="resnet50")
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    datamodule = Datamodule(root="./datasets/kittingRobotV2", train_batch_size=32, eval_batch_size=32, name="kittingRobotV2")

    # Evaluator holds the metrics computed at val and test time separately.
    # AUROC uses continuous scores; F1Score uses thresholded binary labels.
    evaluator = Evaluator(
        val_metrics=[
            AUROC(fields=["pred_score", "gt_label"]),
        ],
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"]),
            F1Score(fields=["pred_label", "gt_label"]),
        ],
    )



    # MLflow logger tracks hyperparameters, metrics and model artifacts per run
    mlflow_logger = AnomalibMLFlowLogger(
        experiment_name="Anomaly Detection",
        run_name=f"{MODEL} kitting robot V2",
        log_model="all",   # save the full model checkpoint as an artifact
        save_dir="./mlruns",
    )

    model = build_model(MODEL, evaluator)

    engine = Engine(
        accelerator="gpu",
        max_epochs=MAX_EPOCHS,
        logger=mlflow_logger,
    )

    # train() fits the model and runs validation; call engine.test() afterwards
    # to get final test metrics on the held-out set.
    engine.train(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
