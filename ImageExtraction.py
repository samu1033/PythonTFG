import cv2
import os
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, frame_prefix="frame"):
    """
    Extract all frames from a video file and save them as individual images.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory where frames will be saved
        frame_prefix (str): Prefix for saved frame filenames (default: "frame")
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame as image file
        frame_filename = os.path.join(output_dir, f"{frame_prefix}_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        
        print(f"Extracted frame {frame_count}")
    
    cap.release()
    print(f"Total frames extracted: {frame_count}")


video_file = r"C:\Users\samuc\Videos\RobotStudio 3-15_2.mp4"
output_folder = r"C:\Users\samuc\Desktop\TFG\PruebasPython\datasets\robotV2\test\anomaly"
    
extract_frames_from_video(video_file, output_folder)