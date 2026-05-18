import cv2
from pathlib import Path

def frames_to_video(frames_dir, output_path="./results/videos/kittingRobot.mp4", fps=15, extension="*.png"):
    """Assemble a sorted sequence of image frames into an MP4 video.

    Args:
        frames_dir:   Directory containing the PNG frames (sorted by filename).
        output_path:  Destination path for the output video file.
        fps:          Playback frame rate of the resulting video.
        extension:    Glob pattern used to select frames (default: *.png).
    """
    frames = sorted(Path(frames_dir).glob(extension))

    if not frames:
        print("No frames found")
        return

    # Read the first frame to get the dimensions
    first = cv2.imread(str(frames[0]))
    h, w, _ = first.shape

    # mp4v is a widely supported codec for .mp4 containers
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
    print(f"Video saved at {output_path} ({len(frames)} frames at {fps} fps)")

# Usage
frames_to_video(
    frames_dir="./results/Padim/kittingRobotDatamodule/v0/images/rgb",
    output_path="./results/videos/kittingRobot.mp4",
    fps=15,
)
