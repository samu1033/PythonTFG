import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow
from torchmetrics.classification import BinaryAUROC


class LSTM(nn.Module):
    def __init__(
        self,
        input_size:  int   = 1,
        hidden_size: int   = 32,
        num_layers:  int   = 1,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.sigmoid(self.fc(last_hidden)).squeeze(-1)


class SequenceDataset(Dataset):
    """Sliding-window dataset over a 1-D anomaly score sequence.

    Each sample is a window of length `seq_len`; its label is the gt_label
    of the last frame in the window.
    """

    def __init__(self, scores: torch.Tensor, labels: torch.Tensor, seq_len: int):
        self.X = scores.unfold(0, seq_len, 1).unsqueeze(-1).float()  # (N, T, 1)
        self.y = labels[seq_len - 1:].float()                         # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_sequence_dataset(predictions, seq_len: int) -> SequenceDataset:
    all_scores = torch.tensor(
        [item.pred_score.item() for batch in predictions for item in batch],
        dtype=torch.float32,
    )
    all_labels = torch.tensor(
        [float(item.gt_label.item()) for batch in predictions for item in batch],
        dtype=torch.float32,
    )
    return SequenceDataset(all_scores, all_labels, seq_len)


def train_lstm(
    predictions,
    seq_len:     int   = 10,
    epochs:      int   = 200,
    lr:          float = 1e-4,
    batch_size:  int   = 32,
    val_split:   float = 0.2,
    hidden_size: int   = 32,
    num_layers:  int   = 1,
    dropout:     float = 0.0,
    checkpoint:  str   = "lstm_best.pth",
) -> tuple[LSTM, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = build_sequence_dataset(predictions, seq_len)
    n_val   = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_seq_ds, val_seq_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_seq_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_seq_ds,   batch_size=batch_size, shuffle=False)

    print(f"Sequences — train: {n_train}  val: {n_val}  (seq_len={seq_len})")

    model     = LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    history       = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_val_loss = float("inf")
    best_epoch    = 1

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("AnomalyDetection")

    with mlflow.start_run(run_name="AnomalyLSTM"):
        mlflow.log_params({
            "model":       "AnomalyLSTM",
            "input_size":  1,
            "hidden_size": hidden_size,
            "num_layers":  num_layers,
            "dropout":     dropout,
            "lr":          lr,
            "epochs":      epochs,
            "batch_size":  batch_size,
            "seq_len":     seq_len,
        })

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss_total = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss  = criterion(preds, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item()
            train_loss_avg = train_loss_total / len(train_loader)

            # Validation
            model.eval()
            val_loss_total = 0.0
            auroc_metric   = BinaryAUROC().to(device)
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    preds           = model(x_batch)
                    val_loss_total += criterion(preds, y_batch).item()
                    auroc_metric.update(preds, y_batch.int())

            val_loss_avg = val_loss_total / len(val_loader)
            val_auroc    = auroc_metric.compute().item()

            history["train_loss"].append(train_loss_avg)
            history["val_loss"].append(val_loss_avg)
            history["val_auroc"].append(val_auroc)

            mlflow.log_metrics(
                {"train_loss": train_loss_avg, "val_loss": val_loss_avg, "val_auroc": val_auroc},
                step=epoch,
            )

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_epoch    = epoch + 1
                torch.save(model.state_dict(), checkpoint)

            print(
                f"Epoch {epoch+1:02d}/{epochs}  "
                f"Train Loss: {train_loss_avg:.4f}  "
                f"Val Loss: {val_loss_avg:.4f}  "
                f"Val AUROC: {val_auroc:.4f}"
            )

        mlflow.log_metrics({
            "best_val_loss":   best_val_loss,
            "best_epoch":      float(best_epoch),
            "final_val_auroc": history["val_auroc"][-1],
        })
        mlflow.log_artifact(checkpoint, artifact_path="checkpoints")
        print(f"\nBest model saved at epoch {best_epoch}  Val Loss: {best_val_loss:.4f}")

    return model, history