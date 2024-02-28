# run it after capturing faces to save encoding,
# and then run faceEncodingsMP.py to run face detection and rec sys
import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np


def compute_face_descriptor(landmarks, image_shape):
    face_descriptor = []
    for lm in landmarks.landmark:
        x, y = int(lm.x * image_shape[1]), int(lm.y * image_shape[0])
        face_descriptor.extend([x, y])
    return np.array(face_descriptor)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

image_dir = "captured_faces/"
data = []

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(image_dir, filename)
        image = cv2.imread(path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # should be only one face per image
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            face_descriptor = compute_face_descriptor(results.multi_face_landmarks[0], image.shape)
            data.append(face_descriptor.tolist())
            # data.append((filename, face_descriptor.tolist()))

# df = pd.DataFrame(data, columns=['filename', 'face_encoding'])
df = pd.DataFrame({'face_encoding': [str(enc) for enc in data]})
df.to_csv('face_encodingsMPtest.csv', index=False)

face_mesh.close()
print("Face encodings have been saved to face_encodingsMPtest.csv")
