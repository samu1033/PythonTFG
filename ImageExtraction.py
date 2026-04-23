import cv2
import os
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, frame_prefix="frame", target_fps=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    step = round(source_fps / target_fps) if target_fps else 1
    print(f"Source FPS: {source_fps:.2f} | Target FPS: {target_fps or source_fps} | Keeping 1 of every {step} frames")

    frame_index = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % step == 0:
            frame_filename = os.path.join(output_dir, f"{frame_prefix}_{saved_count:06d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    print(f"Total frames saved: {saved_count}")


video_file = r"C:\Users\samuc\Videos\RobotStudio 4-22_1.mp4"
output_folder = r"C:\Users\samuc\Desktop\TFG\PythonTFG\datasets\robotV3\test\anomaly"

extract_frames_from_video(video_file, output_folder, target_fps=10)