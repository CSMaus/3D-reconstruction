import cv2
import os
import numpy as np
import time
import torch

video_folder = "D:/ML_DL_AI_stuff/!DoosanSW/Data-20240315T004246Z-001/Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))
video_idx = 2
frame_idx = 0
# need to do:
# open video by frames
# define the welding torch
# define the welding specimen parts (boundaries, etc.)
# define their position in relation to the torch for each frame
# define the movement of the torch in relation to the specimen

# need to show them in the image: (maybe different colors, lines, etc.)
# define position of rod
# define position of root of the weld

# maybe need to add bilateral filtering? or, filter the image based on 5 common frames?
# changes in frames might be noise, BUT the torch is moving, so it might be incorrect (depends on speed)

# cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
# 'd' is the diameter of each pixel neighborhood,
# 'sigmaColor' is the filter sigma in the color space,
# 'sigmaSpace' is the filter sigma in the coordinate space.
d = 9
sigmaColor = 75
sigmaSpace = 75
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


def update_p1(val):
    global update_param1
    update_param1 = max(1, val)


def update_p2(val):
    global update_param2
    update_param2 = max(1, val)


def update_frame_idx(val):
    global frame_idx
    frame_idx = max(0, val)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


cap = cv2.VideoCapture(video_folder + videos[2])
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

edge_buffer = []

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_name = videos[video_idx]
cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
cv2.createTrackbar('Frame_i', video_name, 0, frame_count-1, update_frame_idx)

cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
cv2.createTrackbar('P1', 'Edges', 1, 300, update_p1)
cv2.createTrackbar('P2', 'Edges', 1, 300, update_p2)

cv2.namedWindow('Bilateral', cv2.WINDOW_NORMAL)
cv2.createTrackbar('d', 'Bilateral', 1, 55, update_d)
cv2.createTrackbar('sColor', 'Bilateral', 1, 300, update_sigmaColor)
cv2.createTrackbar('sSpace', 'Bilateral', 1, 300, update_sigmaSpace)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    # to play video normally after seeking comment out
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered_frame = cv2.bilateralFilter(gray_frame, d, sigmaColor, sigmaSpace)
    cv2.imshow('Bilateral', filtered_frame)
    edges = cv2.Canny(filtered_frame, update_param1, update_param2)
    cv2.imshow('Edges', edges)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        color = np.random.randint(0, 255, (3)).tolist()
        cv2.drawContours(frame, contours, i, color, 2)
    cv2.imshow(video_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()



