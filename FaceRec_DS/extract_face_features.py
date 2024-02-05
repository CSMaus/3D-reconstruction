import cv2
import mediapipe as mp
import face_recognition
import pandas as pd
import os

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

image_dir = "captured_faces/"

data = []

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(image_dir, filename)
        image = cv2.imread(path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_detection.process(image_rgb)
        if results.detections:
            # take only one face from one image
            face_locations = face_recognition.face_locations(image_rgb)
            if face_locations:
                face_encoding = face_recognition.face_encodings(image_rgb, known_face_locations=face_locations)[0]
                data.append(("Kseniia", face_encoding.tolist()))
# 坊ちゃん
# botchan

df = pd.DataFrame(data, columns=['filename', 'face_encoding'])
df.to_csv('face_encodings.csv', index=False)

face_detection.close()

print("Face encodings have been saved to face_encodings.csv")
