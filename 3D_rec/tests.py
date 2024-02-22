import cv2

cap = cv2.VideoCapture('test_video.mp4')

if not cap.isOpened():
    print("Error opening video file")
else:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames in the video: {total_frames}")

cap.release()
