import cv2
from pathlib import Path

def frames_to_video(frames_dir, output_path="output.mp4", fps=30, extension="*.png"):
    frames = sorted(Path(frames_dir).glob(extension))
    
    if not frames:
        print("No se encontraron frames")
        return

    # Lee el primer frame para obtener las dimensiones
    first = cv2.imread(str(frames[0]))
    h, w, _ = first.shape

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        writer.write(frame)

    writer.release()
    print(f"✅ Video guardado en {output_path} ({len(frames)} frames a {fps} fps)")

# Uso
frames_to_video(
    frames_dir="./results/Padim/CustomDataModule/v7/images/anomaly",
    output_path="./results/output.mp4",
    fps=30,
)