import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, 20.0, (640, 480))

recording = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)

    if key == ord('r'):  # 'r' to start/stop recording
        recording = not recording
        if recording:
            print("Recording...")
        else:
            print("Recording stopped")

    if recording:
        out.write(frame)

    if key & 0xFF == 27:  # 'Esc' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
