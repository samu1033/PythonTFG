import os
from anomalib.engine import Engine
from anomalib.models import Fastflow, Padim, EfficientAd, Patchcore, Fre
from anomalib.metrics import AUROC, F1Score
from anomalib.metrics.evaluator import Evaluator
from anomalib.loggers import AnomalibMLFlowLogger
from anomalib.visualization import ImageVisualizer

from dataset import Datamodule

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

# --- Config ---
MODEL = "efficientad"  # opciones: "fastflow", "padim", "efficientad", "patchcore"
MAX_EPOCHS = 10
# ---------------------


def build_model(name: str, evaluator: Evaluator):
    if name == "fastflow":
        return Fastflow(backbone="resnet18", evaluator=evaluator)
    if name == "padim":
        return Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"], n_features=200, pre_trained=True, evaluator=evaluator)
    if name == "efficientad":
        return EfficientAd(imagenet_dir="./datasets/imagenette", model_size="small", evaluator=evaluator)
    if name == "patchcore":
        return Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
            evaluator=evaluator,
        )
    if name == "fre":
        return Fre(backbone="resnet50")
    raise ValueError(f"Modelo desconocido: {name}")


def main() -> None:
    datamodule = Datamodule(root="./datasets/grippy", train_batch_size=1, eval_batch_size=32, name="grippyDatamodule")

    evaluator = Evaluator(
        val_metrics=[
            AUROC(fields=["pred_score", "gt_label"]),
        ],
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"]),
            F1Score(fields=["pred_label", "gt_label"]),
        ],
    )
    
    visualizer = ImageVisualizer(
        fields=["image"],
        field_size=(640,480)
    )

    mlflow_logger = AnomalibMLFlowLogger(
        experiment_name="Anomaly Detection",
        run_name=f"{MODEL} grippy robot v2",
        log_model="all",
        save_dir="./mlruns",
    )

    model = build_model(MODEL, evaluator)
    #model.configure_pre_processor(image_size=(640,480))

    engine = Engine(
        accelerator="gpu",
        max_epochs=MAX_EPOCHS,
        logger=mlflow_logger,
    )

    engine.train(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
