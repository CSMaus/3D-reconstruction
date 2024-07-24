import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 2
video_path = os.path.join(video_folder, videos[video_idx])

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0
brightness = 86
contrast = 63
vibrance = 1.1
hue = 0
saturation = 0
lightness = 59
clip_limit = 2.9
tile_grid_size = 22
gray = True


def update_clip_limit(val):
    global clip_limit
    clip_limit = max(1, val) / 10


def update_tile_grid_size(val):
    global tile_grid_size
    tile_grid_size = max(1, val)


def update_frame_idx(val):
    global frame_idx
    frame_idx = max(0, val)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(img)
    return cl


def adjust_lightness(frame, lightness):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lightness_scale = lightness / 100.0

    if lightness_scale > 0:
        v = v.astype(np.float32)
        v = v + (255 - v) * lightness_scale
        v = np.clip(v, 0, 255).astype(np.uint8)
    else:
        v = v.astype(np.float32)
        v = v + v * lightness_scale
        v = np.clip(v, 0, 255).astype(np.uint8)

    adjusted_hsv = cv2.merge([h, s, v])
    adjusted_frame = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    return adjusted_frame


def adjust_lightness_grayscale(frame, lightness):
    lightness_scale = lightness / 100.0

    if lightness_scale > 0:
        frame = frame.astype(np.float32)
        frame = frame + (255 - frame) * lightness_scale
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    else:
        frame = frame.astype(np.float32)
        frame = frame + frame * lightness_scale
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def apply_adjustments(frame):
    global gray

    if gray:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = apply_clahe(gray_frame)

        img = np.int16(gray_frame)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        adjusted_frame = np.uint8(img)
        adjusted_frame = adjust_lightness_grayscale(adjusted_frame, lightness)
        adjusted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_GRAY2BGR)
    else:
        img = np.int16(frame)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        adjusted_frame = adjust_lightness(img, lightness)

        hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        xval = np.arange(0, 256)
        lut = (255 * np.tanh(vibrance * xval / 255) / np.tanh(1) + 0.5).astype(np.uint8)
        s = cv2.LUT(s, lut)

        h = (h.astype(int) + hue) % 180
        h = h.astype(np.uint8)

        s = cv2.add(s, saturation)
        v = cv2.add(v, lightness)

        adjusted_hsv = cv2.merge([h, s, v])

        adjusted_frame = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
        adjusted_frame = adjust_lightness(adjusted_frame, lightness)

    return adjusted_frame


def calculate_histogram(frame):
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    else:
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    return hist


def calculate_brightness_contrast(frame):
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(frame)
    contrast = np.std(frame)
    return brightness, contrast


width = 500
height = 800
video_name = "Video"
control_name = "Controls"
cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(video_name, width, height)

cv2.namedWindow(control_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(control_name, 500, 600)

cv2.createTrackbar('Frame', control_name, 0, frame_count - 1, update_frame_idx)
cv2.createTrackbar('Brightness', control_name, brightness, 100, lambda v: None)
cv2.createTrackbar('Contrast', control_name, contrast, 100, lambda v: None)
cv2.createTrackbar('Mode: 0=Gray, 1=Color', control_name, 0, 1, lambda v: None)
cv2.createTrackbar('CLAHE Clip Limit', control_name, int(clip_limit * 10), 100, update_clip_limit)
cv2.createTrackbar('CLAHE Tile Grid Size', control_name, tile_grid_size, 32, update_tile_grid_size)
cv2.createTrackbar('Lightness', control_name, lightness, 100, lambda v: None)

color_trackbars_created = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    brightness = cv2.getTrackbarPos('Brightness', control_name) - 50
    contrast = cv2.getTrackbarPos('Contrast', control_name) - 50
    gray = cv2.getTrackbarPos('Mode: 0=Gray, 1=Color', control_name) == 0
    lightness = cv2.getTrackbarPos('Lightness', control_name) - 50

    if not gray:
        if not color_trackbars_created:
            cv2.createTrackbar('Vibrance', control_name, 14, 30, lambda v: None)
            cv2.createTrackbar('Hue', control_name, 0, 180, lambda v: None)
            cv2.createTrackbar('Saturation', control_name, 0, 100, lambda v: None)
            color_trackbars_created = True

        vibrance = cv2.getTrackbarPos('Vibrance', control_name) / 10
        hue = cv2.getTrackbarPos('Hue', control_name)
        saturation = cv2.getTrackbarPos('Saturation', control_name) - 50
    else:
        if color_trackbars_created:
            cv2.destroyWindow('Vibrance')
            cv2.destroyWindow('Hue')
            cv2.destroyWindow('Saturation')
            cv2.destroyWindow('Lightness')
            color_trackbars_created = False

    adjusted_frame = apply_adjustments(frame)

    hist = calculate_histogram(adjusted_frame)
    brightness, contrast = calculate_brightness_contrast(adjusted_frame)
    plt.clf()
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Brightness: {brightness:.2f}, Contrast: {contrast:.2f}')
    plt.pause(0.001)

    cv2.imshow(video_name, adjusted_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

    # frame_idx = cv2.getTrackbarPos('Frame', video_name)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

cap.release()
cv2.destroyAllWindows()
