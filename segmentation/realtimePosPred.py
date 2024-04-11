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
# need more data for electrode
pixValThresh = 10


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

    return overlayed_image


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

    if details['was_cropped']:
        mask_weld_resized = np.pad(mask_weld_resized,
                                   ((details['top_crop'], details['bottom_crop']), (0, 0)),
                                   'constant',
                                   constant_values=0)
        mask_electrode_resized = np.pad(mask_electrode_resized,
                                   ((details['top_crop'], details['bottom_crop']), (0, 0)),
                                   'constant',
                                   constant_values=0)

    mask_weld_resized = cv2.resize(mask_weld_resized.astype(np.float32), (details['original_width'], details['original_height']),
                                   interpolation=cv2.INTER_NEAREST)
    mask_electrode_resized = cv2.resize(mask_electrode_resized.astype(np.float32), (details['original_width'], details['original_height']),
                                        interpolation=cv2.INTER_NEAREST)

    overlay_weld = np.zeros_like(frame)
    overlay_electrode = np.zeros_like(frame)
    overlay_weld[mask_weld_resized > 0] = [0, 255, 0]
    overlay_electrode[mask_electrode_resized > 0] = [100, 0, 255]

    overlayed_image = cv2.addWeighted(frame, 1, overlay_weld, 0.5, 0)
    overlayed_image = cv2.addWeighted(overlayed_image, 1, overlay_electrode, 0.5, 0)

    overlayed_image = add_text_based_on_mask(overlayed_image, mask_weld_resized, "Central Weld", False)
    overlayed_image = add_text_based_on_mask(overlayed_image, mask_electrode_resized, "Electrode")

    central_electrode_position = find_central_electrode_position(mask_electrode_resized)
    overlayed_image = annotate_mask_edges_with_position(overlayed_image, mask_weld_resized, central_electrode_position,
                                      details['original_width'])

    cv2.circle(overlayed_image, (details['original_height'], central_electrode_position), 2, (255, 0, 0), -1)
    return overlayed_image


def annotate_mask_edges_with_position(overlayed_image, mask_resized, central_electrode_position, image_width,
                                      step=15):
    """
    :param overlayed_image: Image to draw annotations on
    :param mask_resized: Binary mask of the CentralWeld (0  and 1)
    :param central_electrode_position: Central electrode position (x-coordinate)
    :param image_width: Width of the image for calculating relative positions
    :param step: Step for iterating over the mask rows (to define edge position)
    """
    half_width = image_width // 2
    prev_y = 0
    for y in range(mask_resized.shape[0]):
        if y - prev_y < step and y != 0:
            continue
        row = mask_resized[y, :]

        indices = np.where(row >=0.99)[0]
        # 2420 frame - problem with top right indices
        if indices.size > 0:
            left_edge = min(indices)  # [0]
            right_edge = max(indices)  # [-1]
            if left_edge < half_width:
                left_position = (left_edge - central_electrode_position) / half_width * 100
                cv2.circle(overlayed_image, (left_edge, y), 2, (255, 50, 50), -1)
                cv2.putText(overlayed_image, f"{left_position:.1f}%", (left_edge, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 50, 50), 1)
                prev_y = y

            if right_edge > central_electrode_position:
                right_position = (right_edge - central_electrode_position) / half_width * 100

                cv2.circle(overlayed_image, (right_edge, y), 2, (50, 255, 50), -1)
                cv2.putText(overlayed_image, f"{right_position:.1f}%", (right_edge, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (50, 255, 50), 1)
                prev_y = y
    return overlayed_image


def find_central_electrode_position(mask_resized):
    contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    electrode_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            electrode_contour = contour

    if electrode_contour is not None:
        x, y, w, h = cv2.boundingRect(electrode_contour)
        center_x = x + w // 2
        return center_x

    return mask_resized.shape[1] // 2


# to predict the distance from edge of the weld to the electrode
# need to calculate for each height pixels of mask the rightest and the leftest pixels
# these pixels are the edge of the weld, and write their distance to the electrode as % of the half width of the image
# the center of the electrode by width is the mean of all middle points
# (point between two edges point for each height point)
#


video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 0  # video 1 need to collect more data for all, and 3 too for electrode
frame_idx = 0

num_pixels = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model_weld = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model_weld.classifier[4] = torch.nn.Conv2d(num_pixels, 1, kernel_size=(1, 1), stride=(1, 1))
model_weld.load_state_dict(torch.load('models/CentralWeld-deeplabv3_resnet101-BS32-2024-04-08_14-22.pth'),  # retrained_deeplabv3_resnet101-2024-03-21_13-11.pth'),
                           strict=False)  # , map_location='cpu'
model_weld = model_weld.to(device)
model_weld.eval()

model_electrode = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model_electrode.classifier[4] = torch.nn.Conv2d(num_pixels, 1, kernel_size=(1, 1), stride=(1, 1))
model_electrode.load_state_dict(torch.load('models/Electrode-deeplabv3_resnet101-BS32-2024-04-08_19-44.pth'), strict=False)  # , map_location='cpu'
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

createGif = False

if not createGif:
    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame_i', video_name, 300, frame_count - 1, update_frame_idx)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        # to play video normally after seeking comment out
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

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