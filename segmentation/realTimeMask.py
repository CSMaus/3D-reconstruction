import cv2
import os
import numpy as np
from PIL import Image, ImageOps
import time
import torch
import torchvision
import torchvision.transforms as T


def resize_image(img):
    image_np = np.array(img)
    top_row = np.mean(image_np[0, :, :])
    bottom_row = np.mean(image_np[-1, :, :])
    if top_row < 5 and bottom_row < 5:
        rows = np.where(np.mean(image_np, axis=(1, 2)) > 5)[0]
        if len(rows) > 0:
            if len(rows) < img.width:
                first_row = int((img.height - img.width) / 2)
                last_row = first_row + img.width
                img = img.crop((0, first_row, img.width, last_row))
            else:
                first_row, last_row = rows[0], rows[-1]
                img = img.crop((0, first_row, img.width, last_row))
    else:
        delta_w = img.height - img.width
        delta_h = 0
        padding = (delta_w // 2, delta_h, delta_w - (delta_w // 2), delta_h)
        img = ImageOps.expand(img, padding, fill=0)

    return img


video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 2
frame_idx = 0

num_pixels = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_weld = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model_weld.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
model_weld.load_state_dict(torch.load('models/retrained_deeplabv3_resnet101-2024-03-21_13-11.pth', map_location='cpu'),
                           strict=False)  # , map_location='cpu'
model_weld = model_weld.to(device)
model_weld.eval()

model_electrode = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model_electrode.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
model_electrode.load_state_dict(torch.load('models/Electrode-deeplabv3_resnet101-2024-03-21_14-39.pth',
                                           map_location='cpu'), strict=False)  # , map_location='cpu'
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


cap = cv2.VideoCapture(video_folder + videos[2])
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_name = videos[video_idx]

cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
cv2.createTrackbar('Frame_i', video_name, 0, frame_count-1, update_frame_idx)


def predict_mask(frame):

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
    colored_mask[predicted_mask_electrode_resized > 0] = [0, 0, 255]
    overlayed_image = cv2.addWeighted(original_image_np, 1, colored_mask, 0.5, 0)

    return overlayed_image


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    # to play video normally after seeking comment out
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    frame = predict_mask(frame)
    cv2.imshow(video_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()








