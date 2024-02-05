# this script is to capture images and save them for further face recognition process.
# face_encodings will be extracted to catch the face feature for comparison
import cv2
import os

folder = "captured_faces"
if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)
img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press 'c' to capture", frame)
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed, exit
        print("Escape hit, closing...")
        break
    elif k % 256 == 99:
        # press 'c' to capture image
        img_name = os.path.join(folder, f"image_{img_counter}.png")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()
