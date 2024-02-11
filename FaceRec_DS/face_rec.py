import cv2
import mediapipe as mp
import face_recognition
import pandas as pd
import numpy as np
import time

df = pd.read_csv('face_encodings.csv')
known_face_encodings = [eval(encoding) for encoding in df['face_encoding']]
known_face_names = df['filename'].tolist()

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

prev_time = time.time()
mouth_open = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_locations = [(y, x + w, y + h, x)]
            face_encodings = face_recognition.face_encodings(img_rgb, known_face_locations=face_locations)

            if face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=0.6)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if time.time() - prev_time > 3:
        mesh_results = face_mesh.process(img_rgb)
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                top_lip = face_landmarks.landmark[13].y
                bottom_lip = face_landmarks.landmark[14].y
                mouth_gap = np.abs(top_lip - bottom_lip)

                left_eye_top = face_landmarks.landmark[159].y
                left_eye_bottom = face_landmarks.landmark[145].y
                right_eye_top = face_landmarks.landmark[386].y
                right_eye_bottom = face_landmarks.landmark[374].y

                left_eye_closed = np.abs(left_eye_top - left_eye_bottom) < 0.003
                right_eye_closed = np.abs(right_eye_top - right_eye_bottom) < 0.003

                if left_eye_closed and right_eye_closed:
                    print("Blink detected.")

                if mouth_gap < 0.002:
                    print("Please, unfreeze")

                    # TODO: by eyes blinks
                    # mouth_open = not mouth_open
                    # print(f"Mouth movement detected: {'Open' if mouth_open else 'Closed'}")

        prev_time = time.time()

    mesh_results = face_mesh.process(img_rgb)
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:

            left_eye_top = face_landmarks.landmark[159].y
            left_eye_bottom = face_landmarks.landmark[145].y
            right_eye_top = face_landmarks.landmark[386].y
            right_eye_bottom = face_landmarks.landmark[374].y

            left_eye_closed = np.abs(left_eye_top - left_eye_bottom) < 0.003
            right_eye_closed = np.abs(right_eye_top - right_eye_bottom) < 0.003

            if left_eye_closed and right_eye_closed:
                print("Blink detected.")

    cv2.imshow('Frame', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_detection.close()
face_mesh.close()
