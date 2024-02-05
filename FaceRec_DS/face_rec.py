import cv2
import mediapipe as mp
import face_recognition
import pandas as pd

df = pd.read_csv('face_encodings.csv')
known_face_encodings = [eval(encoding) for encoding in df['face_encoding']]
known_face_names = df['filename'].tolist()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

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

            face_locations = [(y, x+w, y+h, x)]
            face_encodings = face_recognition.face_encodings(img_rgb, known_face_locations=face_locations)

            if face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=0.6)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed, exit
        print("Escape hit, closing...")
        break

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_detection.close()









