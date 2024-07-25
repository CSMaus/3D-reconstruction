import os
import cv2
import numpy as np
import time


def brightness_and_contrast_auto(src, clip_hist_percent=0):
    hist_size = 256
    alpha, beta = 0, 0
    min_gray, max_gray = 0, 0

    if src.ndim == 2:
        gray = src
    elif src.shape[2] == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    elif src.shape[2] == 4:
        gray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image type")

    if clip_hist_percent == 0:
        min_gray, max_gray = np.min(gray), np.max(gray)
    else:
        hist = cv2.calcHist([gray], [0], None, [hist_size], [0, 256])
        hist = hist.flatten()
        accumulator = np.cumsum(hist)

        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        min_gray = 0
        while accumulator[min_gray] < clip_hist_percent:
            min_gray += 1

        max_gray = hist_size - 1
        while accumulator[max_gray] >= (maximum - clip_hist_percent):
            max_gray -= 1

    input_range = max_gray - min_gray

    alpha = (hist_size - 1) / input_range
    beta = -min_gray * alpha

    auto_result = cv2.convertScaleAbs(src, alpha=alpha, beta=beta)

    if src.shape[2] == 4:
        auto_result[:, :, 3] = src[:, :, 3]

    return auto_result


video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 0
video_path = os.path.join(video_folder, videos[video_idx])

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    adjusted_frame = brightness_and_contrast_auto(frame, clip_hist_percent=0.5)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Adjusted Frame", adjusted_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

    time.sleep(1/60)

cap.release()
cv2.destroyAllWindows()
