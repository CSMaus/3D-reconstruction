import cv2
import os
import numpy as np
import time
import torch

video_folder = "Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 1
frame_idx = 250
d = 9
sigmaColor = 75
sigmaSpace = 75
d2 = 9
sigmaColor2 = 75
sigmaSpace2 = 75
update_param1 = 100
update_param2 = 200


def update_d(val):
    global d
    d = max(2, val)


def update_sigmaColor(val):
    global sigmaColor
    sigmaColor = max(1, val)


def update_sigmaSpace(val):
    global sigmaSpace
    sigmaSpace = max(1, val)


def update_d2(val):
    global d2
    d2 = max(2, val)


def update_sigmaColor2(val):
    global sigmaColor2
    sigmaColor2 = max(1, val)


def update_sigmaSpace2(val):
    global sigmaSpace2
    sigmaSpace2 = max(1, val)


def update_p1(val):
    global update_param1
    update_param1 = max(20, val)


def update_p2(val):
    global update_param2
    update_param2 = max(10, val)


def update_beta(val):
    global beta
    beta = max(0, val)


def update_alpha(val):
    global alpha
    alpha = max(0, val)


def update_frame_idx(val):
    global frame_idx
    frame_idx = max(0, val)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


def update_count_len(val):
    global count_len
    count_len = max(1, val)


cap = cv2.VideoCapture(video_folder + videos[video_idx])
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

edge_buffer = []
count_len = 10
alpha = 1
beta = 1

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_name = videos[video_idx]
cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
cv2.createTrackbar('Frame_i', video_name, 0, frame_count, update_frame_idx)
cv2.createTrackbar('Count Len', video_name, 0, 1000, update_count_len)
cv2.createTrackbar('Contrast', video_name, 0, 100, update_beta)
cv2.createTrackbar('Bright', video_name, 0, 30, update_alpha)

cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
cv2.createTrackbar('P1', 'Edges', 20, 300, update_p1)
cv2.createTrackbar('P2', 'Edges', 20, 300, update_p2)

cv2.namedWindow('Bilateral', cv2.WINDOW_NORMAL)
cv2.createTrackbar('d', 'Bilateral', 3, 55, update_d)
cv2.createTrackbar('sColor', 'Bilateral', 1, 300, update_sigmaColor)
cv2.createTrackbar('sSpace', 'Bilateral', 1, 300, update_sigmaSpace)

cv2.namedWindow('BilateralCombo', cv2.WINDOW_NORMAL)
cv2.createTrackbar('d', 'BilateralCombo', 3, 55, update_d2)
cv2.createTrackbar('sColor', 'BilateralCombo', 1, 300, update_sigmaColor2)
cv2.createTrackbar('sSpace', 'BilateralCombo', 1, 300, update_sigmaSpace2)
color = [0, 255, 0]  # np.random.randint(0, 255, (3)).tolist()

cv2.setTrackbarPos('Frame_i', video_name, frame_idx)
cv2.setTrackbarPos('Count Len', video_name, count_len)
cv2.setTrackbarPos('Contrast', video_name, alpha)
cv2.setTrackbarPos('Bright', video_name, beta)
cv2.setTrackbarPos('P1', 'Edges', update_param1)
cv2.setTrackbarPos('P2', 'Edges', update_param2)
cv2.setTrackbarPos('d', 'Bilateral', d)
cv2.setTrackbarPos('sColor', 'Bilateral', sigmaColor)
cv2.setTrackbarPos('sSpace', 'Bilateral', sigmaSpace)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    frame = cv2.convertScaleAbs(frame, alpha=alpha/10, beta=beta)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered_frame = cv2.bilateralFilter(gray_frame, d, sigmaColor, sigmaSpace)
    cv2.imshow('Bilateral', filtered_frame)

    edges = cv2.Canny(filtered_frame, update_param1, update_param2)
    # combined = cv2.addWeighted(filtered_frame, 1, edges, 1, 0)
    # cv2.imshow('Edges', combined)  # edges)

    # combined_filtered = cv2.bilateralFilter(combined, d2, sigmaColor2, sigmaSpace2)
    # cv2.imshow('BilateralCombo', combined_filtered)

    # edges2 = cv2.Canny(combined_filtered, update_param1, update_param2)
    # cv2.imshow('Edges', edges2)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_threshold = 150
    max_area_threshold = 5000
    min_perimeter_threshold = count_len
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > min_perimeter_threshold:  # min_area_threshold < area < max_area_threshold and
            epsilon = 0.001 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            cv2.drawContours(frame, [approx], -1, color, 2)

    cv2.imshow(video_name, frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()



