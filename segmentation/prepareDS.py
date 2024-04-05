import cv2
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import os

pixValThresh = 10
# for 2 of 5 electrode positions also ready
video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
# video_idx = 2
video_name = "Weld_Video_2023-04-20_02-34-42_Camera02.avi.avi"

video_path = video_folder+video_name
xml_path = f'Data/Annotations/{video_name[:-8]}-annotations.xml'

if not os.path.exists(xml_path):
    print(f"Annotations file not found: {xml_path}")
    exit()

name_prefix = video_name[22:-8]

# Weld_Video_2023-04-20_02-19-11_Camera01_annotations.xml'
label_type = 'Electrode'  # 'CentralWeld' 'Electrode'
frames_output_dir = Path(f'SegmentationDS/{label_type}/frames/')
masks_output_dir = Path(f'SegmentationDS/{label_type}/masks/')
imgs_path = Path(f'SegmentationDS/{video_name}/frames/')
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
cap = cv2.VideoCapture(video_folder + video_name)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def check_issave_frame(frame, thresh=pixValThresh):
    save = False
    # if the frame is not almost dark, i e the bottom rows of trhe frame are not almost black
    # i e the max pixels values are above the threshold, then we don't save the frame
    max_pixel_values = np.mean(frame, axis=(1, 2))
    bottom_crop = 0
    while max_pixel_values[-(bottom_crop + 1)] < thresh and (frame_height - bottom_crop) > frame_width:
        bottom_crop += 1
    if bottom_crop > 0:
        save = True

    return save


# this for loop works for annotation which were made using video
num_saved_imgs = 0
for track in root.findall(".//track"):
    label = track.get('label')
    if label == label_type:
        for polygon in track.findall(".//polygon"):
            frame_index = int(polygon.get('frame'))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at index {frame_index}")
                continue
            if not check_issave_frame(frame):
                continue

            points_str = polygon.get('points')
            points = parse_points(points_str)
            mask = generate_mask((frame_height, frame_width, 1), points)

            frame_path = frames_output_dir / f"frame_{name_prefix}_{str(frame_index)}.jpg"
            mask_path = masks_output_dir / f"mask_{name_prefix}_{str(frame_index)}.png"
            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(mask_path), mask)

            # flip frame and mask along vertical axis
            frame = cv2.flip(frame, 1)
            mask = cv2.flip(mask, 1)
            frame_path = frames_output_dir / f"frameF_{name_prefix}_{str(frame_index)}.jpg"
            mask_path = masks_output_dir / f"maskF_{name_prefix}_{str(frame_index)}.png"
            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(mask_path), mask)

            num_saved_imgs += 2
cap.release()


if num_saved_imgs == 0:
    print("No track found in the xml file. Using second approach")
    # here is for loop for annotation which were made using images
    for image_tag in root.findall('.//image'):
        image_id = image_tag.get('id')
        image_name = image_tag.get('name')
        frame_index = image_name[6:-4]

        frame = cv2.imread(os.path.join(imgs_path, image_name))

        if frame is None:
            print(f"Failed to read image: {imgs_path+image_name}")
            continue
            
        if not check_issave_frame(frame):
            continue

        for polygon in image_tag.findall('.//polygon'):
            label = polygon.get('label')
            if label == label_type:
                points_str = polygon.get('points')
                points = parse_points(points_str)

                mask = generate_mask((frame_height, frame_width, 1), points)

                frame_path = frames_output_dir / f"frame_{name_prefix}_{frame_index}.jpg"
                mask_path = masks_output_dir / f"mask_{name_prefix}_{frame_index}.png"
                cv2.imwrite(str(frame_path), frame)
                cv2.imwrite(str(mask_path), mask)

                # flip frame and mask along vertical axis
                frame = cv2.flip(frame, 1)
                mask = cv2.flip(mask, 1)
                frame_path = frames_output_dir / f"frameF_{name_prefix}_{str(frame_index)}.jpg"
                mask_path = masks_output_dir / f"maskF_{name_prefix}_{str(frame_index)}.png"
                cv2.imwrite(str(frame_path), frame)
                cv2.imwrite(str(mask_path), mask)

                num_saved_imgs += 2
print("Saved images: ", num_saved_imgs)
print("Processing completed.")

