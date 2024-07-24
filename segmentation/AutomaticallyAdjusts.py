import os
import cv2
import numpy as np

video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 3
video_path = os.path.join(video_folder, videos[video_idx])

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


def apply_clahe(img, clip_limit=5.0, tile_grid_size=(20, 20)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(img)
    return cl


def automatic_brightness_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


while True:
    ret, frame = cap.read()
    if not ret:
        break

    adjusted_frame = automatic_brightness_contrast(frame)

    # Convert to grayscale for CLAHE
    gray_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)
    clahe_frame = apply_clahe(gray_frame)

    # Convert back to BGR for display
    clahe_frame_bgr = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Adjusted Frame", clahe_frame_bgr)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
