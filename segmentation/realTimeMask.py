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
from datetime import datetime
thresh = 10


# need to collect data for "Weld_Video_2023-04-20_01-55-23_Camera02.avi"
pixValThresh = 10


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


'''def predict_mask_v2(frame, thresh=pixValThresh, isShowImages=False):
    """
    I'll add it later
    """
    image = Image.fromarray(frame).convert("RGB")
    processed_image, details = preprocess_image_for_prediction(image, thresh)

    if isShowImages:
        # save original and processed images
        if not os.path.exists('test'):
            os.makedirs('test')
        processed_image.save('test/processed_image.jpg')
        image.save('test/original_image.jpg')

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
'''


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


def preprocess_image_for_prediction(img, thresh=pixValThresh, desired_size=256):
    """
    Adjusts the image by cropping rows from bottom and/or top if the maximum pixel value in the row is below the threshold,
    and if needed, pads the image to make it square before resizing to the desired size.
    """
    image_np = np.array(img)
    max_pixel_values = np.mean(image_np, axis=(1, 2))
    height, width = image_np.shape[:2]

    bottom_crop = 0
    while max_pixel_values[-(bottom_crop + 1)] < thresh and (height - bottom_crop) > width:
        bottom_crop += 1

    top_crop = 0
    while max_pixel_values[top_crop] < thresh and (height - bottom_crop - top_crop) > width:
        top_crop += 1

    if bottom_crop > 0 or top_crop > 0:
        img = img.crop((0, top_crop, width, height - bottom_crop))
        was_cropped = True
    else:
        was_cropped = False

    new_height, new_width = img.size[1], img.size[0]

    if new_height > new_width:
        padding = ((new_height - new_width) // 2, 0)
        img = ImageOps.expand(img, (padding[0], 0, new_height - new_width - padding[0], 0), fill=0)
        was_padded = True
    else:
        was_padded = False

    if img.size[0] == img.size[1]:
        img = img.resize((desired_size, desired_size), Image.LANCZOS)

    crop_or_pad_details = {
        'top_crop': top_crop,
        'bottom_crop': bottom_crop,
        'was_cropped': was_cropped,
        'was_padded': was_padded,
        'original_height': height,
        'original_width': width
    }

    return img, crop_or_pad_details


def add_text_based_on_mask(overlayed_image, mask_resized, text, is_electrode=True):
    contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        text_color = (100, 0, 255)
        if is_electrode:
            text_position = (x + w + 5, y + h // 2)
        else:
            text_position = (x - 10, y + h + 20)
            text_color = (0, 255, 0)

        text_position = (max(0, text_position[0]), max(0, text_position[1]))

        cv2.putText(overlayed_image, text,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    text_color, 1, cv2.LINE_AA)
        break
    return overlayed_image


def smooth_mask_edges(mask):
    mask = (mask > 0).astype(np.uint8) * 255

    cont = cv2.bitwise_not(mask)
    original = np.zeros_like(mask)
    smoothed = np.full_like(mask, 255, dtype=np.uint8)
    filter_radius = 5
    filter_size = 2 * filter_radius + 1
    sigma = 10

    contours, hierarchy = cv2.findContours(cont, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for j in range(len(contours)):
        length = len(contours[j]) + 2 * filter_radius
        idx = (len(contours[j]) - filter_radius)
        x = []
        y = []
        for i in range(length):
            x.append(contours[j][(idx + i) % len(contours[j])][0][0])
            y.append(contours[j][(idx + i) % len(contours[j])][0][1])

        x_filt = cv2.GaussianBlur(np.array(x, dtype=np.float32), (filter_size, filter_size), sigma, sigma)
        y_filt = cv2.GaussianBlur(np.array(y, dtype=np.float32), (filter_size, filter_size), sigma, sigma)

        smooth_contours = []
        smooth = []
        for i in range(filter_radius, len(contours[j]) + filter_radius):
            smooth.append([int(x_filt[i]), int(y_filt[i])])
        smooth_contours.append(np.array(smooth, dtype=np.int32))

        color = (0, 0, 0) if hierarchy[0][j][3] < 0 else (1, 1, 1)

        cv2.drawContours(smoothed, [np.array(smooth_contours)], 0, color, thickness=cv2.FILLED)

    return smoothed


def enhance_outer_contour(image, mask, color=(0, 255, 0), thickO=2):
    # mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_overlay = np.zeros_like(image, dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(contour_overlay, [contour], -1, color, thickO)
    enhanced_image = cv2.addWeighted(image, 1, contour_overlay, 0.5, 0)

    return enhanced_image


def predict_mask(frame, thresh=pixValThresh, isShowImages=False):
    """
    TODO: fix. Sometime it moves to the side incorrectly (when there is no welding and cropping from bottom, i e when padding was made)
    Fix1: works well if the image was only padded
    Fix 2: okaaay, now is better, but sometimes (with worse lighting) it still moves to the top or bottom
    """
    image = Image.fromarray(frame).convert("RGB")
    processed_image, details = preprocess_image_for_prediction(image, thresh)

    if isShowImages:
        if not os.path.exists('test'):
            os.makedirs('test')
        processed_image.save('test/processed_image1.jpg')
        image.save('test/original_image1.jpg')

    image_tensor = transform(processed_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_weld = model_weld(image_tensor)['out'][0][0]
        output_electrode = model_electrode(image_tensor)['out'][0][0]
        predicted_mask_weld = output_weld.sigmoid().cpu().numpy() > 0.5
        predicted_mask_electrode = output_electrode.sigmoid().cpu().numpy() > 0.5

    cropped_height = details['original_height'] - details['top_crop'] - details['bottom_crop']
    if details['was_padded']:
        cropped_width = cropped_height
    else:
        cropped_width = details['original_width']

    mask_weld_resized = cv2.resize(predicted_mask_weld.astype(np.float32), (cropped_width, cropped_height),
                                   interpolation=cv2.INTER_NEAREST)
    mask_electrode_resized = cv2.resize(predicted_mask_electrode.astype(np.float32), (cropped_width, cropped_height),
                                        interpolation=cv2.INTER_NEAREST)

    if details['was_padded']:
        # crop the mask width to the original width by half from left and right sides
        # maybe need to change  original_height and original_width to cropped_height and cropped_width
        diff = int((cropped_height - details['original_width'])/2)
        mask_weld_resized = mask_weld_resized[:, diff:cropped_height - diff]
        mask_electrode_resized = mask_electrode_resized[:, diff:cropped_height - diff]

    # if the image was cropped, make padding to top and bottom as shown in the details
    if details['was_cropped']:
        mask_weld_resized = np.pad(mask_weld_resized,
                                   ((details['top_crop'], details['bottom_crop']), (0, 0)),
                                   'constant',
                                   constant_values=0)
        mask_electrode_resized = np.pad(mask_electrode_resized,
                                   ((details['top_crop'], details['bottom_crop']), (0, 0)),
                                   'constant',
                                   constant_values=0)

    canvas_weld = np.zeros((details['original_height'], details['original_width']), dtype=np.float32)
    canvas_electrode = np.zeros((details['original_height'], details['original_width']), dtype=np.float32)

    vertical_start = details['top_crop']
    vertical_end = details['original_height'] - details['bottom_crop']

    mask_weld_resized = cv2.resize(mask_weld_resized.astype(np.float32), (details['original_width'], details['original_height']),
                                   interpolation=cv2.INTER_NEAREST)
    mask_electrode_resized = cv2.resize(mask_electrode_resized.astype(np.float32), (details['original_width'], details['original_height']),
                                        interpolation=cv2.INTER_AREA)

    mask_electrode_resized = smooth_mask_edges(mask_electrode_resized)
    # canvas_weld[vertical_start:vertical_end, :] = mask_weld_resized
    # canvas_electrode[vertical_start:vertical_end, :] = mask_electrode_resized

    # canvas_weld = (canvas_weld * 255).astype(np.uint8)
    # canvas_electrode = (canvas_electrode * 255).astype(np.uint8)

    overlay_weld = np.zeros_like(frame)
    overlay_electrode = np.zeros_like(frame)
    overlay_weld[mask_weld_resized > 0] = [0, 255, 0]
    overlay_electrode[mask_electrode_resized > 0] = [100, 0, 255]

    # overlayed_image = cv2.addWeighted(frame, 1, overlay_weld, 0.5, 0)
    overlayed_image = cv2.addWeighted(frame, 1, overlay_electrode, 0.5, 0)  # was overlayed_image instead of frame

    bright_mask = create_brightest_mask(frame, mask_electrode_resized)
    bright_overlay = np.zeros_like(frame)
    bright_overlay[bright_mask > 0] = [0, 255, 255]
    overlayed_image = cv2.addWeighted(overlayed_image, 1, bright_overlay, 0.8, 0)
    overlayed_image = enhance_outer_contour(overlayed_image, mask_electrode_resized, color=[100, 0, 255], thickO=2)
    overlayed_image = enhance_outer_contour(overlayed_image, bright_mask, color=[0, 255, 255], thickO=2)

    # overlayed_image = add_text_based_on_mask(overlayed_image, mask_weld_resized, "Central Weld", False)
    overlayed_image = add_text_based_on_mask(overlayed_image, bright_mask, "Arc", False)
    overlayed_image = add_text_based_on_mask(overlayed_image, mask_electrode_resized, "Electrode")

    return overlayed_image


def create_brightest_mask(frame, electrode_mask, base_threshold=180, adjust_factor=0.5, max_distance=15):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_frame)

    adjusted_threshold = max(0, base_threshold - int(adjust_factor * mean_brightness))

    bright_mask = (gray_frame >= adjusted_threshold).astype(np.uint8)

    electrode_mask_resized = cv2.resize(electrode_mask, (bright_mask.shape[1], bright_mask.shape[0])).astype(np.uint8)

    distance_transform = cv2.distanceTransform(1 - electrode_mask_resized, cv2.DIST_L2, 5)

    distance_mask = (distance_transform <= max_distance).astype(np.uint8)

    filtered_bright_mask = cv2.bitwise_and(bright_mask, distance_mask)

    final_mask = cv2.bitwise_and(filtered_bright_mask, 1 - electrode_mask_resized)

    return final_mask


def predict_mask_v2(frame, thresh=pixValThresh, isShowImages=False):
    """
    Modified function to include creation and overlay of the brightest part mask.
    """
    image = Image.fromarray(frame).convert("RGB")
    processed_image, details = preprocess_image_for_prediction(image, thresh)

    if isShowImages:
        # save original and processed images
        if not os.path.exists('test'):
            os.makedirs('test')
        processed_image.save('test/processed_image.jpg')
        image.save('test/original_image.jpg')

    image_tensor = transform(processed_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_weld = model_weld(image_tensor)['out'][0][0]
        output_electrode = model_electrode(image_tensor)['out'][0][0]
        predicted_mask_weld = output_weld.sigmoid().cpu().numpy() > 0.5
        predicted_mask_electrode = output_electrode.sigmoid().cpu().numpy() > 0.5

    processed_dims = (processed_image.width, processed_image.height)
    mask_weld_resized = cv2.resize(predicted_mask_weld.astype(np.float32), processed_dims,
                                   interpolation=cv2.INTER_NEAREST)
    mask_electrode_resized = cv2.resize(predicted_mask_electrode.astype(np.float32), processed_dims,
                                        interpolation=cv2.INTER_NEAREST)

    original_height, original_width = details['original_height'], details['original_width']

    canvas_weld = np.zeros((original_height, original_width), dtype=np.float32)
    canvas_electrode = np.zeros((original_height, original_width), dtype=np.float32)

    if details['was_cropped']:
        vertical_pad_top = details['top_index']
        vertical_pad_bottom = original_height - details['bottom_index']
        canvas_weld[vertical_pad_top:original_height - vertical_pad_bottom, :] = mask_weld_resized
        canvas_electrode[vertical_pad_top:original_height - vertical_pad_bottom, :] = mask_electrode_resized
    if details['was_padded']:
        horizontal_pad = (processed_image.width - original_width) // 2
        canvas_weld[:, :] = mask_weld_resized[:, horizontal_pad:horizontal_pad + original_width]
        canvas_electrode[:, :] = mask_electrode_resized[:, horizontal_pad:horizontal_pad + original_width]

    canvas_weld = (canvas_weld * 255).astype(np.uint8)
    canvas_electrode = (canvas_electrode * 255).astype(np.uint8)

    overlay_weld = np.zeros_like(frame)
    overlay_electrode = np.zeros_like(frame)
    overlay_weld[canvas_weld > 0] = [0, 255, 0]
    overlay_electrode[canvas_electrode > 0] = [255, 0, 255]

    # overlayed_image = cv2.addWeighted(frame, 1, overlay_weld, 0.5, 0)
    overlayed_image = cv2.addWeighted(frame, 1, overlay_electrode, 0.5, 0)  # was overlayed_image instead of frame

    bright_mask = create_brightest_mask(frame, canvas_electrode, brightness_threshold=200)
    bright_overlay = np.zeros_like(frame)
    bright_overlay[bright_mask > 0] = [0, 255, 0]
    overlayed_image = cv2.addWeighted(overlayed_image, 1, bright_overlay, 0.5, 0)

    return overlayed_image


video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 2  # video 1 need to collect more data for all, and 3 too for electrode
frame_idx = 0
createGif = False

num_pixels = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model_weld = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model_weld.classifier[4] = torch.nn.Conv2d(num_pixels, 1, kernel_size=(1, 1), stride=(1, 1))
model_weld.load_state_dict(torch.load('models/CentralWeld-deeplabv3_resnet101-BS32-2024-04-08_14-22.pth'),
                           # retrained_deeplabv3_resnet101-2024-03-21_13-11.pth'),
                           strict=False)  # , map_location='cpu'


model_weld = model_weld.to(device)
model_weld.eval()

model_electrode = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model_electrode.classifier[4] = torch.nn.Conv2d(num_pixels, 1, kernel_size=(1, 1), stride=(1, 1))
model_electrode.load_state_dict(torch.load('models/Electrode-deeplabv3_resnet101-BS32-2024-07-21_19-05.pth'),
                                strict=False)
# , map_location='cpu'
# 'Electrode-deeplabv3_resnet101-BS32-2024-04-08_19-44.pth'  # this one is better.
# TODO: Prepare mask only for the low part of the electrode - iot should be higher accuracy


# 'models/Electrode-deeplabv3_resnet101-BS32-2024-04-08_19-44.pth' works well for video 0, 2,
#
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

frame_counter = 0
font = cv2.FONT_HERSHEY_SIMPLEX
x, y = 10, 500
position = (x, y)
fontScale = 0.7
fontColor = (245, 245, 245)
thickness = 2
lineType = 2


if not createGif:
    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame_i', video_name, 0, frame_count - 1, update_frame_idx)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        # to play video normally after seeking comment out
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # frame_counter += 1
        frame_counter = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # if frame_counter % 50 == 0:
        processed_frame = predict_mask(frame)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        cv2.putText(processed_frame, f"Frame idx:",
                    position,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        cv2.putText(processed_frame, f"{frame_counter}",
                    (x, y + int(fontScale * 35)),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        # if frame_counter == -1:
        #     processed_frame = predict_mask(frame, pixValThresh, True)
        #     print("Frame processed")
        # frames_for_gif.append(processed_frame_rgb)
        cv2.imshow(video_name, processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    # sys.exit()
else:
    frames_for_gif = []
    print(f"Preparing gif for {video_name} ...")
    for frame_idx in tqdm(range(289, frame_count - 310, 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame_counter = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        processed_frame = predict_mask(frame)
        # processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        cv2.putText(processed_frame, f"Frame idx:",
                    position,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        cv2.putText(processed_frame, f"{frame_counter}",
                    (x, y + int(fontScale * 35)),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        frames_for_gif.append(processed_frame)

    date = datetime.today().strftime('%Y-%m-%d_%H-%M')
    video_path = os.path.join("Videos/", f'{video_name[:-8]}-{date}.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
    height, width, _ = frames_for_gif[0].shape
    out = cv2.VideoWriter(video_path, fourcc, 15.0, (width, height))
    for frame in frames_for_gif:
        out.write(frame)
    out.release()
    # gif_path = os.path.join("Gifs/", f'{video_name[:-8]}-{date}.gif')
    # imageio.mimsave(gif_path, frames_for_gif, fps=10)
    print("video saved at: ", video_path)

cap.release()
cv2.destroyAllWindows()
# sys.exit()








