import cv2
from pathlib import Path
import os
import tqdm

video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 2
video_name = videos[video_idx]
video_path = video_folder+video_name
frames_output_dir = Path(f'SegmentationDS/{video_name}/frames')
print("Making frames for video: ", video_name, "...")
# import sys
# sys.exit()
if not os.path.exists(frames_output_dir):
    frames_output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_folder + videos[video_idx])
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Number of frames: ", frame_count, "...")

each_N_frame = 10

for i in tqdm.tqdm(range(0, frame_count, each_N_frame)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame at index {i}")
        continue

    frame_path = frames_output_dir / f"frame_{i:04d}.jpg"
    cv2.imwrite(str(frame_path), frame)

cap.release()
print('Frames saved!')

