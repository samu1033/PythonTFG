import os
import mlflow
import mlflow.pytorch

from anomalib.engine import Engine
from anomalib.models import Fastflow, Padim, EfficientAd, Patchcore
from anomalib.metrics import AUROC, F1Score
from anomalib.metrics.evaluator import Evaluator
from anomalib.loggers import AnomalibMLFlowLogger
from anomalib.visualization import ImageVisualizer

from dataset import kittingRobotDatamodule

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.pytorch.autolog(log_models=True, log_every_n_epoch=1, silent=True)

# --- Configuración ---
MODEL = "padim"  # opciones: "fastflow", "padim", "efficientad", "patchcore"
MAX_EPOCHS = 50
# ---------------------


def build_model(name: str, evaluator: Evaluator, visualizer: ImageVisualizer):
    if name == "fastflow":
        return Fastflow(backbone="resnet18", evaluator=evaluator)
    if name == "padim":
        return Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"], pre_trained=True, evaluator=evaluator, visualizer=visualizer)
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
    raise ValueError(f"Modelo desconocido: {name}")


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
    
    visualizer = ImageVisualizer(
        fields=["image"],
        field_size=(640,480)
    )

    mlflow_logger = AnomalibMLFlowLogger(
        experiment_name="Anomaly Detection",
        run_name=f"{MODEL} real robot",
        log_model="all",
        save_dir="./mlruns",
    )

    model = build_model(MODEL, evaluator, visualizer)
    #model.configure_pre_processor(image_size=(640,480))

    engine = Engine(
        accelerator="gpu",
        max_epochs=MAX_EPOCHS,
        #logger=mlflow_logger,
    )

    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
