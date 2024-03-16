import cv2
import os
import numpy as np
import time
import torch

video_folder = "D:/ML_DL_AI_stuff/!DoosanSW/Data-20240315T004246Z-001/Data/Weld_VIdeo/"
videos = os.listdir(os.path.join(video_folder))


for videoF in videos:
    cap = cv2.VideoCapture(video_folder + videoF)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {videoF} has {frame_count} frames")
    cap.release()
