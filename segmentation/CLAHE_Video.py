import os
import cv2
import numpy as np

# take a look here: https://www.youtube.com/watch?v=FKjk4WFOBrc
# in video approach is much better, but here code is only first attempt

video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 2
video_path = os.path.join(video_folder, videos[video_idx])
width = 400
height = 1400
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0
brightness = 65
contrast = 28
vibrance = 1.5
hue = 5
saturation = 29
lightness = 29
clip_limit = 5.0
tile_grid_size = 12


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
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def apply_adjustments(frame):

    frame = apply_clahe(frame)
    img = np.int16(frame)
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


width = 500
height = 800
video_name = "Video"
control_name = "Controls"
cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(video_name, width, height)

cv2.namedWindow(control_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(control_name, 500, 600)
cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(video_name, width, height)
cv2.createTrackbar('Frame', control_name, 0, frame_count - 1, update_frame_idx)
cv2.createTrackbar('Brightness', control_name, brightness, 100, lambda v: None)
cv2.createTrackbar('Contrast', control_name, contrast, 100, lambda v: None)
cv2.createTrackbar('Vibrance', control_name, int(vibrance*10), 30, lambda v: None)  # vibrance * 10 for trackbar
cv2.createTrackbar('Hue', control_name, hue, 180, lambda v: None)
cv2.createTrackbar('Saturation', control_name, saturation, 100, lambda v: None)
cv2.createTrackbar('Lightness', control_name, lightness, 100, lambda v: None)
cv2.createTrackbar('CLAHE Clip Limit', control_name, int(clip_limit * 10), 100, update_clip_limit)
cv2.createTrackbar('CLAHE Tile Grid Size', control_name, tile_grid_size, 42, update_tile_grid_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    brightness = cv2.getTrackbarPos('Brightness', control_name) - 50
    contrast = cv2.getTrackbarPos('Contrast', control_name) - 50
    vibrance = cv2.getTrackbarPos('Vibrance', control_name) / 10
    hue = cv2.getTrackbarPos('Hue', control_name)
    saturation = cv2.getTrackbarPos('Saturation', control_name) - 50
    lightness = cv2.getTrackbarPos('Lightness', control_name) - 50

    adjusted_frame = apply_adjustments(frame)

    cv2.imshow(video_name, adjusted_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

    # frame_idx = cv2.getTrackbarPos('Frame', video_name)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

cap.release()
cv2.destroyAllWindows()
