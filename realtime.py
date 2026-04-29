import numpy as np
import cv2
import torch
import pyrealsense2 as rs  # type: ignore

from anomalib.models import Padim

# --- Config ---
CHECKPOINT = "./results/Padim/kittingRobotDatamodule/v0/weights/lightning/model.ckpt"
THRESHOLD  = 0.5   # anomaly score 0-1
ALPHA      = 0.5   # heatmap blend opacity
COLORMAP   = cv2.COLORMAP_JET
# --------------


def load_model(checkpoint: str) -> Padim:
    model = Padim.load_from_checkpoint(checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def frame_to_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR uint8 HWC → float CHW in [0,1] with batch dim."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)  # (1, 3, H, W)


def build_overlay(
    frame_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    score: float,
    threshold: float,
    fps: float,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    heat = cv2.resize(anomaly_map, (w, h))
    heat_norm = np.clip((heat - heat.min()) / (heat.max() - heat.min() + 1e-8), 0, 1)
    heat_u8 = (heat_norm * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heat_u8, COLORMAP)
    blended = cv2.addWeighted(frame_bgr, 1 - ALPHA, heatmap_bgr, ALPHA, 0)

    label = "ANOMALY" if score >= threshold else "NORMAL"
    color = (0, 0, 255) if score >= threshold else (0, 255, 0)
    cv2.putText(blended, f"{label}  score={score:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(blended, f"threshold={threshold:.2f}  FPS={fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(blended, "q=quit  +/-=threshold", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return blended


def main() -> None:
    print("Loading PaDiM model …")
    model = load_model(CHECKPOINT)
    device = next(model.parameters()).device
    print(f"Model on {device}")

    pipeline = rs.pipeline()  # type: ignore[attr-defined]
    cfg = rs.config()         # type: ignore[attr-defined]
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # type: ignore[attr-defined]
    pipeline.start(cfg)
    print("RealSense started. q=quit  +/-=threshold")

    threshold = THRESHOLD
    tick_freq = cv2.getTickFrequency()
    prev_tick = cv2.getTickCount()
    fps = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            curr_tick = cv2.getTickCount()
            fps = tick_freq / (curr_tick - prev_tick)
            prev_tick = curr_tick

            frame = np.asanyarray(color_frame.get_data())
            tensor = frame_to_tensor(frame, device)

            with torch.no_grad():
                output = model(tensor)

            score = float(output.pred_score.squeeze().cpu())
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()  # (H, W)

            overlay = build_overlay(frame, anomaly_map, score, threshold, fps)
            cv2.imshow("PaDiM Real-Time Anomaly Detection", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('+'), ord('=')):
                threshold = min(threshold + 0.05, 1.0)
                print(f"Threshold: {threshold:.2f}")
            elif key == ord('-'):
                threshold = max(threshold - 0.05, 0.0)
                print(f"Threshold: {threshold:.2f}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
