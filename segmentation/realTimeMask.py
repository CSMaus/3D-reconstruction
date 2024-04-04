import sys

import cv2
import os
import numpy as np
from PIL import Image, ImageOps
import time
import torch
import torchvision
import torchvision.transforms as T
import imageio
from tqdm import tqdm
thresh = 2

# need to collect data for "Weld_Video_2023-04-20_01-55-23_Camera02.avi"


def resize_image(img):
    image_np = np.array(img)
    top_row = np.mean(image_np[0, :, :])
    bottom_row = np.mean(image_np[-1, :, :])
    '''if top_row < thresh and bottom_row < thresh:
        rows = np.where(np.mean(image_np, axis=(1, 2)) > thresh)[0]
        if len(rows) > 0:
            if len(rows) < img.width:
                first_row = int((img.height - img.width) / 2)
                last_row = first_row + img.width
                img = img.crop((0, first_row, img.width, last_row))
            else:
                first_row, last_row = rows[0], rows[-1]
                img = img.crop((0, first_row, img.width, last_row))
    else:'''
    delta_w = img.height - img.width
    delta_h = 0
    padding = (delta_w // 2, delta_h, delta_w - (delta_w // 2), delta_h)
    img = ImageOps.expand(img, padding, fill=0)

    return img


def preprocess_image_for_prediction(img, thresh=10):
    """
    Adjusts the image to make it square by cropping or padding, similar to the preprocessing during training.
    Returns the processed image and the cropping or padding details.
    """
    image_np = np.array(img)

    rows_to_consider = np.max(image_np, axis=(1, 2)) < thresh
    top_index, bottom_index = 0, len(rows_to_consider) - 1

    while bottom_index > top_index and rows_to_consider[bottom_index] and image_np.shape[0] - (
            bottom_index - top_index + 1) >= image_np.shape[1]:
        bottom_index -= 1

    while top_index < bottom_index and rows_to_consider[top_index] and image_np.shape[0] - (
            bottom_index - top_index + 1) >= image_np.shape[1]:
        top_index += 1

    crop_or_pad_details = {'top_index': top_index, 'bottom_index': bottom_index + 1, 'original_height': img.height, 'original_width': img.width, 'was_cropped': False, 'was_padded': False}

    if top_index > 0 or bottom_index < len(rows_to_consider) - 1:
        img = img.crop((0, top_index, img.width, bottom_index + 1))
        crop_or_pad_details['was_cropped'] = True
    else:
        delta_w = abs(img.width - img.height)
        if img.width < img.height:
            padding = (delta_w // 2, 0, delta_w - (delta_w // 2), 0)
        else:
            padding = (0, delta_w // 2, 0, delta_w - (delta_w // 2))
        img = ImageOps.expand(img, padding, fill=0)
        crop_or_pad_details['padding'] = padding
        crop_or_pad_details['was_padded'] = True

    return img, crop_or_pad_details


video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 1  # video 1 need to collect more data for all, and 3 too for electrode
frame_idx = 0

num_pixels = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model_weld = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model_weld.classifier[4] = torch.nn.Conv2d(num_pixels, 1, kernel_size=(1, 1), stride=(1, 1))
model_weld.load_state_dict(torch.load('models/CentralWeld-deeplabv3_resnet101-2024-04-03_15-32.pth'),  # retrained_deeplabv3_resnet101-2024-03-21_13-11.pth'),
                           strict=False)  # , map_location='cpu'
model_weld = model_weld.to(device)
model_weld.eval()

model_electrode = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model_electrode.classifier[4] = torch.nn.Conv2d(num_pixels, 1, kernel_size=(1, 1), stride=(1, 1))
model_electrode.load_state_dict(torch.load('models/Electrode-deeplabv3_resnet101-2024-04-03_17-35.pth'), strict=False)  # , map_location='cpu'
model_electrode = model_electrode.to(device)
model_electrode.eval()
transform = T.Compose([
    T.Resize((num_pixels, num_pixels)),
    T.ToTensor(),
])


def update_frame_idx(val):
    global frame_idx
    frame_idx = max(0, val)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


cap = cv2.VideoCapture(video_folder + videos[video_idx])
if not cap.isOpened():
    print("Video end")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_name = videos[video_idx]


def predict_mask(frame, thresh=10):
    """
    I'll add it later
    """
    image = Image.fromarray(frame).convert("RGB")
    processed_image, details = preprocess_image_for_prediction(image, thresh)

    image_tensor = transform(processed_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_weld = model_weld(image_tensor)['out'][0][0]
        output_electrode = model_electrode(image_tensor)['out'][0][0]
        predicted_mask_weld = output_weld.sigmoid().cpu().numpy() > 0.5
        predicted_mask_electrode = output_electrode.sigmoid().cpu().numpy() > 0.5

    processed_dims = (processed_image.width, processed_image.height)
    mask_weld_resized = cv2.resize(predicted_mask_weld.astype(np.float32), processed_dims, interpolation=cv2.INTER_NEAREST)
    mask_electrode_resized = cv2.resize(predicted_mask_electrode.astype(np.float32), processed_dims, interpolation=cv2.INTER_NEAREST)

    original_height, original_width = details['original_height'], details['original_width']

    canvas_weld = np.zeros((original_height, original_width), dtype=np.float32)
    canvas_electrode = np.zeros((original_height, original_width), dtype=np.float32)

    if details['was_cropped']:
        vertical_pad_top = details['top_index']
        vertical_pad_bottom = original_height - details['bottom_index']
        canvas_weld[vertical_pad_top:original_height-vertical_pad_bottom, :] = mask_weld_resized
        canvas_electrode[vertical_pad_top:original_height-vertical_pad_bottom, :] = mask_electrode_resized
    if details['was_padded']:
        horizontal_pad = (processed_image.width - original_width) // 2
        canvas_weld[:, :] = mask_weld_resized[:, horizontal_pad:horizontal_pad+original_width]
        canvas_electrode[:, :] = mask_electrode_resized[:, horizontal_pad:horizontal_pad+original_width]

    canvas_weld = (canvas_weld * 255).astype(np.uint8)
    canvas_electrode = (canvas_electrode * 255).astype(np.uint8)

    overlay_weld = np.zeros_like(frame)
    overlay_electrode = np.zeros_like(frame)
    overlay_weld[canvas_weld > 0] = [0, 255, 0]
    overlay_electrode[canvas_electrode > 0] = [255, 0, 255]

    overlayed_image = cv2.addWeighted(frame, 1, overlay_weld, 0.5, 0)
    overlayed_image = cv2.addWeighted(overlayed_image, 1, overlay_electrode, 0.5, 0)

    return overlayed_image


def predict_mask_old(frame):

    image = Image.fromarray(frame).convert("RGB")
    # image = Image.open(input_image_path).convert("RGB")
    image = resize_image(image)
    original_image_np = np.array(image)
    original_size = image.size
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model_weld(image)['out']
        probs = torch.sigmoid(output)
        predicted_mask_weld = (probs > 0.5).float()

        output2 = model_electrode(image)['out']
        probs2 = torch.sigmoid(output2)
        predicted_mask_electrode = (probs2 > 0.5).float()

    predicted_mask_weld_np = predicted_mask_weld.cpu().numpy().squeeze(0).squeeze(0)
    predicted_mask_weld_resized = cv2.resize(predicted_mask_weld_np, (original_size[0], original_size[1]),
                                            interpolation=cv2.INTER_NEAREST)
    predicted_mask_weld_resized = (predicted_mask_weld_resized * 255).astype(np.uint8)

    predicted_mask_electrode_np = predicted_mask_electrode.cpu().numpy().squeeze(0).squeeze(0)
    predicted_mask_electrode_resized = cv2.resize(predicted_mask_electrode_np, (original_size[0], original_size[1]),
                                            interpolation=cv2.INTER_NEAREST)
    predicted_mask_electrode_resized = (predicted_mask_electrode_resized * 255).astype(np.uint8)

    colored_mask = np.zeros_like(original_image_np)
    colored_mask[predicted_mask_weld_resized > 0] = [0, 255, 0]
    colored_mask[predicted_mask_electrode_resized > 0] = [255, 0, 255]
    overlayed_image = cv2.addWeighted(original_image_np, 1, colored_mask, 0.5, 0)

    return overlayed_image


frame_counter = 0

cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
cv2.createTrackbar('Frame_i', video_name, 0, frame_count-1, update_frame_idx)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    # to play video normally after seeking comment out
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    frame_counter += 1
    # if frame_counter % 50 == 0:
    processed_frame = predict_mask(frame)
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    # frames_for_gif.append(processed_frame_rgb)
    cv2.imshow(video_name, processed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
sys.exit()


frames_for_gif = []
frame_counter = 0
print(f"Preparing gif for {video_name} ...")
for frame_idx in tqdm(range(0, frame_count, 10)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    processed_frame = predict_mask(frame)
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    frames_for_gif.append(processed_frame_rgb)


gif_path = os.path.join("Gifs/", f'{video_name[:-8]}.gif')
imageio.mimsave(gif_path, frames_for_gif, fps=4)
print("gif saved at: ", gif_path)

cap.release()
cv2.destroyAllWindows()
sys.exit()








