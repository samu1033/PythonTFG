<!-- Repo-specific Copilot instructions for AI coding agents -->
# Copilot instructions — Pruebas Python (Transistor)

Purpose: orient an AI coding agent to be immediately productive in this repository (anomalib PatchCore experiments over the MVTec "transistor" category).

**Big Picture**
- **What this repo does:** runs Patchcore anomaly-detection experiments (anomalib) against the MVTec-style `transistor` dataset. The runner is `main.py`.
- **Data flow:** the code uses anomalib's `MVTecAD` data loader which expects a root that contains a subfolder named `transistor` with the MVTec layout: `train/good`, `test/<classes>`, `ground_truth/<classes>`.
- **Results:** model outputs are written under `resultados_transistor` (and there are mirrors in `results/`). Experiment checkpoints and visual outputs live under `resultados_transistor/Patchcore/...`.

**Key files and places to look**
- `main.py` — primary experiment runner. It shows required config fields and important Windows notes (use `num_workers=0`).
- `Dataset/transistor/` — dataset root; must follow MVTec folder layout.
- `resultados_transistor/` and `results/` — where Patchcore writes checkpoints, predictions and figures.

**Important Conventions & Patterns**
- Set `RUTA_ABSOLUTA` in `main.py` to the parent folder that contains the `transistor` directory (do NOT include the `transistor` segment itself). Example: `RUTA_ABSOLUTA = r"C:\Users\samuc\Desktop\TFG\Dataset"`.
- The code intentionally uses `MVTecAD` (not a generic Folder loader). Use category names matching the dataset folder name (here: `transistor`).
- On Windows, keep `num_workers=0` in `MVTecAD` to avoid multiprocessing file-loader issues.
- Batch sizes are tuned for a GTX 1650 in `main.py` (`train_batch_size=4`). Reduce further on memory errors.
- The training loop uses anomalib's `Engine` with `engine.fit(...)` and `engine.test(...)` — changing `default_root_dir` will move where results/checkpoints are stored.

**Install / Run (discovered from code / environment)**
- Minimal dependency hints (install in virtualenv):

```
pip install torch torchvision
pip install anomalib
```

- Run the experiment from repository root:

```
python main.py
```

**What to change for new experiments**
- To run a different MVTec category, point `MVTecAD(..., category="<name>")` and ensure `Dataset/<name>/` exists with MVTec layout.
- To change where artifacts go, set `default_root_dir` when creating `Engine(...)` or move the `resultados_transistor` folder.

**Troubleshooting and notes**
- If dataloader errors occur on Windows, confirm `num_workers=0` and that `RUTA_ABSOLUTA` is a valid absolute path.
- If CUDA is not available, anomalib/torch will fall back to CPU if `accelerator="auto"` is used; lower `batch_size` accordingly.
- Check `resultados_transistor/Patchcore/...` after runs for checkpoints and visual outputs.

**Example snippets (from repo)**
- Dataset loader (from `main.py`):

```
datamodule = MVTecAD(root=RUTA_ABSOLUTA, category="transistor", train_batch_size=4, eval_batch_size=4, num_workers=0)
```

- Engine setup (from `main.py`):

```
engine = Engine(accelerator="auto", devices=1, max_epochs=1, default_root_dir="./resultados_transistor")
```

If anything in these instructions is unclear or you want more detail about dependency versions, CI, or where to store long-term experiment artifacts, tell me which parts to expand. 
