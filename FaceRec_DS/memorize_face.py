import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
import time


def compute_face_descriptor(landmarks, image_shape):
    face_descriptor = []
    for lm in landmarks.landmark:
        x, y = int(lm.x * image_shape[1]), int(lm.y * image_shape[0])
        face_descriptor.extend([x, y])
    return np.array(face_descriptor)


def compare_faces(known_encodings, face_encoding):
    if len(known_encodings) == 0:
        return True
    face_encoding_norm = face_encoding / np.linalg.norm(face_encoding)
    known_encodings_norm = known_encodings / np.linalg.norm(known_encodings, axis=1, keepdims=True)
    cosine_similarity = np.dot(known_encodings_norm, face_encoding_norm)
    distances = 1 - cosine_similarity
    return distances[np.argmin(distances)] * 1000


def compare_faces_eu(known_encodings, face_encoding):
    if len(known_encodings) == 0:
        return 10000
    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    return distances[np.argmin(distances)]


def update_time(val):
    global update_interval
    update_interval = max(5, val)


cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 800, 700)
cv2.createTrackbar('Update t', 'Frame', 5, 300, update_time)

update_interval = 5

face_encodings_file = 'face_encodingsMPauto.csv'
if os.path.exists(face_encodings_file):
    df = pd.read_csv(face_encodings_file)
    known_encodings = [np.array(eval(enc)) for enc in df['face_encoding']]
else:
    known_encodings = []

folder = "captured_faces/"
threshold = 0.2
threshold_eu = 350
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
img_counter = len(os.listdir(os.path.join(folder)))
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - prev_time >= update_interval / 100:
        cv2.imshow("Frame", frame)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_descriptor = compute_face_descriptor(results.multi_face_landmarks[0], frame.shape)

            # compare_result = compare_faces(known_encodings, face_descriptor)
            compare_result_eu = compare_faces_eu(known_encodings, face_descriptor)

            if len(known_encodings) == 0 or compare_result_eu > threshold_eu:
                img_name = os.path.join(folder, f"image_{img_counter}.png")
                cv2.imwrite(img_name, frame)
                print(f"{img_name} saved")
                img_counter += 1

                known_encodings.append(face_descriptor.tolist())
        prev_time = current_time

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC pressed
        break
cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame({'face_encoding': [str(enc) for enc in known_encodings]})
df.to_csv(face_encodings_file, index=False)

face_mesh.close()

print(f"Face encodings have been saved to {face_encodings_file}")
