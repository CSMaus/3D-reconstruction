import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time


def compute_face_descriptor(landmarks, image_shape):
    # simple landmarks descriptor
    face_descriptor = []
    for lm in landmarks.landmark:
        x, y = int(lm.x * image_shape[1]), int(lm.y * image_shape[0])
        face_descriptor.extend([x, y])
    return np.array(face_descriptor)


def compare_faces(known_encodings, face_encoding):
    if len(known_encodings) == 0:
        return None
    # Euclidean distance
    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    return np.argmin(distances), distances[np.argmin(distances)]


def update_time(val):
    global update_interval
    update_interval = max(1, val)


def update_blur_width(val):
    global blur_width
    blur_width = val


update_interval = 10  # *0.1 sec
blur_width = 50
cv2.namedWindow('Face rec tests', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face rec tests', 800, 700)
cv2.createTrackbar('Blur Width', 'Face rec tests', 10, 390, update_blur_width)
cv2.createTrackbar('Update t', 'Face rec tests', 1, 300, update_time)

df = pd.read_csv('face_encodingsMP.csv')
known_face_encodings = [np.array(eval(encoding)) for encoding in df['face_encoding']]
known_face_names = df['filename'].tolist()

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
prev_time = time.time()
face_info = []
name = "Recognized person name"
print("Press Esc to close the programm")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_time = time.time()
    if current_time - prev_time >= update_interval / 100:
        face_info = []
        results = face_detection.process(img_rgb)
        prev_time = current_time

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_results = face_mesh.process(img_rgb)
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        face_descriptor = compute_face_descriptor(face_landmarks, frame.shape)
                        match_index, distance = compare_faces(known_face_encodings, face_descriptor)
                        if distance < 450:
                            name = "Kseniia"  #known_face_names[match_index]
                        else:
                            print(distance)
                            name = "Unknown"

                        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    face_info.append((x, y, w, h, name))
    else:
        for (x, y, w, h, name) in face_info:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    frame[:, :blur_width] = cv2.blur(frame[:, :blur_width], (blur_width, blur_width))
    frame[:, -blur_width:] = cv2.blur(frame[:, -blur_width:], (blur_width, blur_width))

    cv2.imshow('Face rec tests', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_detection.close()
face_mesh.close()
