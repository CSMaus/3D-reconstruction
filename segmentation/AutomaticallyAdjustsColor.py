import os
import cv2
import numpy as np

video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 1
video_path = os.path.join(video_folder, videos[video_idx])

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


def apply_clahe_color(img, clip_limit=5.4, tile_grid_size=(12, 12)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def automatic_brightness_contrast(image, clip_hist_percent=1, brightness_boost=0, contrast_boost=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = (255 / (maximum_gray - minimum_gray)) * contrast_boost
    beta = -minimum_gray * alpha + brightness_boost

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


while True:
    ret, frame = cap.read()
    if not ret:
        break

    adjusted_frame = automatic_brightness_contrast(frame, brightness_boost=10, contrast_boost=1)
    clahe_frame = apply_clahe_color(adjusted_frame)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Adjusted Frame", clahe_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
