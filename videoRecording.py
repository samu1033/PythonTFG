"""Record synchronized color and depth frames from a RealSense D435 camera.

Press SPACE to toggle recording on/off; press 'q' to quit.
Color frames are saved as standard BGR PNGs.
Depth frames are saved as 16-bit PNGs (values in mm) to preserve the raw
metric depth — lossy formats like JPEG would destroy the depth precision.
"""
import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path

# ── Output folders ───────────────────────────────────────────────────────────
color_dir = Path("datasets/kittingRobotV2/train/good/")
depth_dir = Path("datasets/kittingRobotV2/depth/train/good")
color_dir.mkdir(parents=True, exist_ok=True)
depth_dir.mkdir(parents=True, exist_ok=True)

# ── Camera configuration ─────────────────────────────────────────────────────
FPS = 15  # The D435 only supports: 6, 15, 30 or 60 fps

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, FPS)
pipeline.start(config)

frame_count = 0
recording = False

print("Press SPACE to start/stop recording. Press 'q' to quit.")
print(f"  Color     -> {color_dir}")
print(f"  Depth     -> {depth_dir}")

def draw_ui(combined, recording, frame_count):
    """Overlay recording status and controls onto the preview frame."""
    h, w = combined.shape[:2]

    # Bottom-center button: green when idle, red when recording
    btn_text  = "[ SPACE ] STOP" if recording else "[ SPACE ] RECORD"
    btn_color = (0, 0, 200) if recording else (0, 180, 0)
    cv2.rectangle(combined, (w//2 - 140, h - 45), (w//2 + 140, h - 10), btn_color, -1)
    cv2.putText(combined, btn_text, (w//2 - 125, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if recording:
        # Blinking red dot + frame counter in the top-left corner
        cv2.circle(combined, (18, 18), 9, (0, 0, 255), -1)
        cv2.putText(combined, f"REC  {frame_count} frames", (35, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(combined, "WAITING", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    return combined

try:
    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())           # uint16 in mm (to save)

        # ── Save frames only if recording ────────────────────────────────────
        if recording:
            name = f"frame_{frame_count:05d}.png"
            cv2.imwrite(str(color_dir / name), color_img)
            # PNG is lossless and supports 16-bit, so depth values are preserved exactly
            cv2.imwrite(str(depth_dir / name), depth_raw)
            frame_count += 1

        # ── Real-time visualization ───────────────────────────────────────────
        combined = color_img.copy()
        combined = draw_ui(combined, recording, frame_count)
        cv2.imshow("RealSense  |  Color (left)  -  Depth (right)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            recording = not recording
            state = "STARTED" if recording else "STOPPED"
            print(f"Recording {state}. Frames saved so far: {frame_count}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\nSession ended. {frame_count} frames saved.")
