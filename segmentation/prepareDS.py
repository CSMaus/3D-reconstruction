import cv2
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import os

video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 2
video_name = "Weld_Video_2023-04-20_02-19-11_Camera01.avi.avi"

video_path = video_folder+video_name
xml_path = 'Data/Annotations/Weld_Video_2023-04-20_02-19-11_Camera01_annotations.xml'
frames_output_dir = Path('SegmentationDS/frames')
masks_output_dir = Path('SegmentationDS/masks')
frames_output_dir.mkdir(parents=True, exist_ok=True)
masks_output_dir.mkdir(parents=True, exist_ok=True)


def parse_points(points_str):
    return [tuple(map(float, point.split(','))) for point in points_str.split(';')]


def generate_mask(shape, points):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


tree = ET.parse(xml_path)
root = tree.getroot()
cap = cv2.VideoCapture(video_folder + videos[video_idx])
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

for track in root.findall(".//track"):
    label = track.get('label')
    if label == 'CentralWeld':
        for polygon in track.findall(".//polygon"):
            frame_index = int(polygon.get('frame'))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at index {frame_index}")
                continue

            points_str = polygon.get('points')
            points = parse_points(points_str)
            mask = generate_mask((frame_height, frame_width, 1), points)

            frame_path = frames_output_dir / f"frame_{frame_index:04d}.jpg"
            mask_path = masks_output_dir / f"mask_{frame_index:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(mask_path), mask)

cap.release()
print("Processing completed.")

