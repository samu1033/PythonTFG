"""Real-time anomaly detection using a PaDiM model and an Intel RealSense D435.

The pipeline runs at ~20 fps:
  1. Grab a color frame from the camera.
  2. Convert it to a normalised tensor and run PaDiM inference.
  3. Render a two-panel overlay (anomaly map + predicted mask) with OpenCV.

Press '+'/'-' to adjust the decision threshold at runtime; press 'q' to quit.
"""
import numpy as np
import cv2
import torch
import pyrealsense2 as rs  # type: ignore

from anomalib.models import Padim

# --- Config ---
CHECKPOINT = "./results/Padim/kittingRobotV2/v8/weights/lightning/model.ckpt"
THRESHOLD  = 0.5    # anomaly score 0-1 (label decision only)
MAP_ALPHA  = 0.5    # image + anomaly map blend (anomalib default)
MASK_COLOR = (0, 0, 255)  # pred mask contour, BGR (red, like anomalib)
DISPLAY_MAX_WIDTH = 1280  # downscale the panel grid to fit the screen
# --------------



def load_model(checkpoint: str) -> Padim:
    """Load a PaDiM checkpoint and move it to GPU if available."""
    model = Padim.load_from_checkpoint(checkpoint, weights_only=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def frame_to_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR uint8 HWC → float CHW in [0,1] with batch dim."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # anomalib models expect values in [0, 1], not [0, 255]
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    return t.unsqueeze(0).to(device)  # (1, 3, H, W)


def _label(img: np.ndarray, text: str, org: tuple[int, int],
           scale: float = 0.7, color=(255, 255, 255)) -> None:
    """White text on a semi-transparent black box."""
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    # Darken the background region so the text is readable on any image
    box = img[max(0, y - th - 6):y + bl, x - 3:x + tw + 3]
    if box.size:
        box[:] = (box * 0.5).astype(np.uint8)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


def build_overlay(
    frame_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    pred_mask: np.ndarray,
    score: float,
    threshold: float,
    fps: float,
) -> np.ndarray:
    """Build a two-panel side-by-side overlay for display.

    Panel 1 — image blended with the jet-colormap anomaly heatmap.
    Panel 2 — image with the predicted anomaly mask drawn as a red contour.


    """
    h, w = frame_bgr.shape[:2]

    # Panel 1: anomaly map blended over the color frame.
    # The map is already normalised by anomalib's post-processor, so we use
    # absolute values — stable and comparable across frames.
    amap_u8 = np.clip(anomaly_map * 255, 0, 255).astype(np.uint8)
    amap_u8 = cv2.resize(amap_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
    panel_map = cv2.addWeighted(frame_bgr, 1 - MAP_ALPHA, heat, MAP_ALPHA, 0)

    # Panel 2: contour of the predicted binary mask drawn over the color frame.
    mask_u8 = (pred_mask.astype(np.uint8) * 255)
    if mask_u8.shape != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
    panel_mask = frame_bgr.copy()
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel_mask, cnts, -1, MASK_COLOR, 2)

    # Titles + live status bar on panel 1.
    _label(panel_map, "Image + Anomaly Map", (10, 28))
    _label(panel_mask, "Image + Pred Mask", (10, 28))
    lbl = "ANOMALY" if score >= threshold else "NORMAL"
    status = f"{lbl}  score={score:.3f}  thr={threshold:.2f}  FPS={fps:.1f}"
    # Red text for anomalies, green for normal frames
    sc = (0, 0, 255) if score >= threshold else (0, 255, 0)
    _label(panel_map, status, (10, h - 12), scale=0.6, color=sc)

    return np.hstack([panel_map, panel_mask])


def main() -> None:
    print("Loading PaDiM model …")
    model = load_model(CHECKPOINT)
    device = next(model.parameters()).device
    print(f"Model on {device}")

    # Configure the RealSense pipeline for color-only streaming
    pipeline = rs.pipeline()  
    cfg = rs.config()         
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  
    pipeline.start(cfg)
    print("RealSense started. q=quit  +/-=threshold")

    threshold = THRESHOLD
    # Use cv2 tick counter for accurate per-frame FPS measurement
    tick_freq = cv2.getTickFrequency()
    prev_tick = cv2.getTickCount()
    fps = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Measure elapsed time between consecutive frames
            curr_tick = cv2.getTickCount()
            fps = tick_freq / (curr_tick - prev_tick)
            prev_tick = curr_tick

            frame = np.asanyarray(color_frame.get_data())
            tensor = frame_to_tensor(frame, device)

            # Disable gradient computation — inference only, saves memory and time
            with torch.no_grad():
                output = model(tensor)

            score = float(output.pred_score.squeeze().cpu())
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()  # (H, W)
            pred_mask = output.pred_mask.squeeze().cpu().numpy().astype(bool)

            overlay = build_overlay(frame, anomaly_map, pred_mask,
                                    score, threshold, fps)

            # Downscale the combined panel if it exceeds the display width
            if overlay.shape[1] > DISPLAY_MAX_WIDTH:
                scale = DISPLAY_MAX_WIDTH / overlay.shape[1]
                overlay = cv2.resize(overlay, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
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
