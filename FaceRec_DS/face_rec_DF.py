from deepface import DeepFace

import cv2
import pandas as pd
from deepface.models.FacialRecognition import FacialRecognition
import numpy as np

# okay, so here is better to use mediapipe for face detection and taing it from image,
# and then use DeepFace for age, gender, emotions

# this part of defining emotions will be used later for AI twin

models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
# so, example iof how to use with video - realtime.py impementation. insted of path use numpy array
# DeepFace.stream(db_path="captured_faces/", model_name=models[3])

db_path = "captured_faces/"

model: FacialRecognition = DeepFace.build_model(model_name=models[3])

detector_backend = "opencv"
distance_metric = "cosine"
enable_face_analysis = True
source = 0
time_threshold = 5
frame_threshold = 5

text_color = (255, 255, 255)
pivot_img_size = 112  # face recognition result image

enable_emotion = True
enable_age_gender = True


# find custom values for this input set
target_size = model.input_shape
if enable_face_analysis:
    DeepFace.build_model(model_name="Age")
    print("Age model is just built")
    DeepFace.build_model(model_name="Gender")
    print("Gender model is just built")
    DeepFace.build_model(model_name="Emotion")
    print("Emotion model is just built")
# -----------------------
# call a dummy find function for db_path once to create embeddings in the initialization
DeepFace.find(
    img_path=np.zeros([224, 224, 3]),
    db_path=db_path,
    model_name=models[3],
    detector_backend=detector_backend,
    distance_metric=distance_metric,
    enforce_detection=False,
)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raw_img = img.copy()
    resolution_x = img.shape[1]
    resolution_y = img.shape[0]
    face_included_frames = 0

    try:
        # just extract the regions to highlight in webcam
        face_objs = DeepFace.extract_faces(
            img_path=img,
            target_size=target_size,
            detector_backend=detector_backend,
            enforce_detection=False,
        )
        faces = []
        for face_obj in face_objs:
            facial_area = face_obj["facial_area"]
            faces.append(
                (
                    facial_area["x"],
                    facial_area["y"],
                    facial_area["w"],
                    facial_area["h"],
                )
            )
    except:  # to avoid exception if no face detected
        faces = []

    if len(faces) == 0:
        face_included_frames = 0
    detected_faces = []
    face_index = 0
    for x, y, w, h in faces:
        if w > 130:  # discard small detected faces

            face_detected = True
            if face_index == 0:
                face_included_frames = (
                        face_included_frames + 1
                )  # increase frame for a single face

            cv2.rectangle(
                img, (x, y), (x + w, y + h), (255, 67, 67), 3
            )

            detected_face = img[int(y): int(y + h), int(x): int(x + w)]  # crop detected face
            detected_faces.append((x, y, w, h))
            face_index = face_index + 1

    cv2.imshow('Frame', img)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed, exit
        print("Escape hit, closing...")
        break

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
