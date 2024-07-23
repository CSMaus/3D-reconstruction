import os
import cv2
import numpy as np

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


def apply_adjustments(frame):
    global gray

    img = np.int16(frame)
    img = np.clip(img + lightness, 0, 255)
    img = np.uint8(img)

    if gray:
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_frame = apply_clahe(gray_frame)

        img = np.int16(gray_frame)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        adjusted_frame = np.uint8(img)
        adjusted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_GRAY2BGR)
    else:
        img = np.int16(img)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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

    return adjusted_frame


width = 400
height = 1300
video_name = "Video"
cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(video_name, width, height)
cv2.createTrackbar('Frame', video_name, 0, frame_count - 1, update_frame_idx)
cv2.createTrackbar('Brightness', video_name, brightness, 100, lambda v: None)
cv2.createTrackbar('Contrast', video_name, contrast, 100, lambda v: None)
cv2.createTrackbar('Mode: 0=Gray, 1=Color', video_name, 0, 1, lambda v: None)
cv2.createTrackbar('CLAHE Clip Limit', video_name, int(clip_limit * 10), 100, update_clip_limit)
cv2.createTrackbar('CLAHE Tile Grid Size', video_name, tile_grid_size, 32, update_tile_grid_size)
cv2.createTrackbar('Lightness', video_name, lightness, 100, lambda v: None)

color_trackbars_created = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    brightness = cv2.getTrackbarPos('Brightness', video_name) - 50
    contrast = cv2.getTrackbarPos('Contrast', video_name) - 50
    gray = cv2.getTrackbarPos('Mode: 0=Gray, 1=Color', video_name) == 0
    lightness = cv2.getTrackbarPos('Lightness', video_name) - 50

    if not gray:
        if not color_trackbars_created:
            cv2.createTrackbar('Vibrance', video_name, 14, 30, lambda v: None)
            cv2.createTrackbar('Hue', video_name, 0, 180, lambda v: None)
            cv2.createTrackbar('Saturation', video_name, 0, 100, lambda v: None)
            color_trackbars_created = True

        vibrance = cv2.getTrackbarPos('Vibrance', video_name) / 10
        hue = cv2.getTrackbarPos('Hue', video_name)
        saturation = cv2.getTrackbarPos('Saturation', video_name) - 50
    else:
        if color_trackbars_created:
            cv2.destroyWindow('Vibrance')
            cv2.destroyWindow('Hue')
            cv2.destroyWindow('Saturation')
            cv2.destroyWindow('Lightness')
            color_trackbars_created = False

    adjusted_frame = apply_adjustments(frame)

    cv2.imshow(video_name, adjusted_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

    # frame_idx = cv2.getTrackbarPos('Frame', video_name)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

cap.release()
cv2.destroyAllWindows()
