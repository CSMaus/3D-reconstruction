import cv2
import mediapipe as mp
import face_recognition
import pandas as pd
import time


# bit GUI for video not to freeze
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
cv2.createTrackbar('Update t', 'Face rec tests', 1, 20, update_time)

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
firstrun = True

face_info = []
name = "Recognized person name"

while cap.isOpened():
    # start_fd = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_time = time.time()
    if current_time - prev_time >= update_interval/10:
        face_info = []
        results = face_detection.process(img_rgb)
        prev_time = current_time
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_locations = [(y, x + w, y + h, x)]
                face_encodings = face_recognition.face_encodings(img_rgb, known_face_locations=face_locations)

                # start_fr = time.time()
                if face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=0.6)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                face_info.append((x, y, w, h, name))
    else:
        for (x, y, w, h, name) in face_info:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    frame[:, :blur_width] = cv2.blur(frame[:, :blur_width], (blur_width, blur_width))
    frame[:, -blur_width:] = cv2.blur(frame[:, -blur_width:], (blur_width, blur_width))

    # leave it for later, now need to upgrade speed
    '''
    if time.time() - prev_time > 3:
        # start_fl = time.time()
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
        # end_fl = time.time()
        # print(f"Face Landmarks Operation Time: {end_fl - start_fl} seconds")



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
    '''
    cv2.imshow('Face rec tests', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    # end_fd = time.time()
    # print(f"All operations time: {end_fd - start_fd} seconds")
    # print("To close the programm press key 'Esc'")

cap.release()
cv2.destroyAllWindows()
face_detection.close()
face_mesh.close()
