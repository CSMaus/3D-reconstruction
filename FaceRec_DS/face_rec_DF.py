from deepface import DeepFace
import mediapipe as mp
import cv2
import pandas as pd
from deepface.models.FacialRecognition import FacialRecognition
import numpy as np

# okay, so here is better to use mediapipe for face detection and taing it from image,
# and then use DeepFace for age, gender, emotions

# this part of defining emotions will be used later for AI twin

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

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

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
                img, (x, y), (x + w, y + h), (67, 67, 255), 3
            )

            detected_face = img[int(y): int(y + h), int(x): int(x + w)]  # crop detected face
            detected_faces.append((x, y, w, h))
            face_index = face_index + 1

            if enable_face_analysis == True:

                demographies = DeepFace.analyze(
                    img_path=custom_face,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                    silent=True,
                )

                if len(demographies) > 0:
                    # directly access 1st face cos img is extracted already
                    demography = demographies[0]

                    if enable_emotion:
                        emotion = demography["emotion"]
                        emotion_df = pd.DataFrame(
                            emotion.items(), columns=["emotion", "score"]
                        )
                        emotion_df = emotion_df.sort_values(
                            by=["score"], ascending=False
                        ).reset_index(drop=True)

                        # background of mood box

                        # transparency
                        overlay = freeze_img.copy()
                        opacity = 0.4

                        if x + w + pivot_img_size < resolution_x:
                            # right
                            cv2.rectangle(
                                freeze_img
                                # , (x+w,y+20)
                                ,
                                (x + w, y),
                                (x + w + pivot_img_size, y + h),
                                (64, 64, 64),
                                cv2.FILLED,
                            )

                            cv2.addWeighted(
                                overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img
                            )

                        elif x - pivot_img_size > 0:
                            # left
                            cv2.rectangle(
                                freeze_img
                                # , (x-pivot_img_size,y+20)
                                ,
                                (x - pivot_img_size, y),
                                (x, y + h),
                                (64, 64, 64),
                                cv2.FILLED,
                            )

                            cv2.addWeighted(
                                overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img
                            )

                        for index, instance in emotion_df.iterrows():
                            current_emotion = instance["emotion"]
                            emotion_label = f"{current_emotion} "
                            emotion_score = instance["score"] / 100

                            bar_x = 35  # this is the size if an emotion is 100%
                            bar_x = int(bar_x * emotion_score)

                            if x + w + pivot_img_size < resolution_x:

                                text_location_y = y + 20 + (index + 1) * 20
                                text_location_x = x + w

                                if text_location_y < y + h:
                                    cv2.putText(
                                        freeze_img,
                                        emotion_label,
                                        (text_location_x, text_location_y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (255, 255, 255),
                                        1,
                                    )

                                    cv2.rectangle(
                                        freeze_img,
                                        (x + w + 70, y + 13 + (index + 1) * 20),
                                        (
                                            x + w + 70 + bar_x,
                                            y + 13 + (index + 1) * 20 + 5,
                                        ),
                                        (255, 255, 255),
                                        cv2.FILLED,
                                    )

                            elif x - pivot_img_size > 0:

                                text_location_y = y + 20 + (index + 1) * 20
                                text_location_x = x - pivot_img_size

                                if text_location_y <= y + h:
                                    cv2.putText(
                                        freeze_img,
                                        emotion_label,
                                        (text_location_x, text_location_y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (255, 255, 255),
                                        1,
                                    )

                                    cv2.rectangle(
                                        freeze_img,
                                        (
                                            x - pivot_img_size + 70,
                                            y + 13 + (index + 1) * 20,
                                        ),
                                        (
                                            x - pivot_img_size + 70 + bar_x,
                                            y + 13 + (index + 1) * 20 + 5,
                                        ),
                                        (255, 255, 255),
                                        cv2.FILLED,
                                    )

                    if enable_age_gender:
                        apparent_age = demography["age"]
                        dominant_gender = demography["dominant_gender"]
                        gender = "M" if dominant_gender == "Man" else "W"
                        analysis_report = str(int(apparent_age)) + " " + gender

                        # -------------------------------

                        info_box_color = (46, 200, 255)

                        # top
                        if y - pivot_img_size + int(pivot_img_size / 5) > 0:

                            triangle_coordinates = np.array(
                                [
                                    (x + int(w / 2), y),
                                    (
                                        x + int(w / 2) - int(w / 10),
                                        y - int(pivot_img_size / 3),
                                    ),
                                    (
                                        x + int(w / 2) + int(w / 10),
                                        y - int(pivot_img_size / 3),
                                    ),
                                ]
                            )

                            cv2.drawContours(
                                freeze_img,
                                [triangle_coordinates],
                                0,
                                info_box_color,
                                -1,
                            )

                            cv2.rectangle(
                                freeze_img,
                                (
                                    x + int(w / 5),
                                    y - pivot_img_size + int(pivot_img_size / 5),
                                ),
                                (x + w - int(w / 5), y - int(pivot_img_size / 3)),
                                info_box_color,
                                cv2.FILLED,
                            )

                            cv2.putText(
                                freeze_img,
                                analysis_report,
                                (x + int(w / 3.5), y - int(pivot_img_size / 2.1)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 111, 255),
                                2,
                            )

                        # bottom
                        elif (
                                y + h + pivot_img_size - int(pivot_img_size / 5)
                                < resolution_y
                        ):

                            triangle_coordinates = np.array(
                                [
                                    (x + int(w / 2), y + h),
                                    (
                                        x + int(w / 2) - int(w / 10),
                                        y + h + int(pivot_img_size / 3),
                                    ),
                                    (
                                        x + int(w / 2) + int(w / 10),
                                        y + h + int(pivot_img_size / 3),
                                    ),
                                ]
                            )

                            cv2.drawContours(
                                freeze_img,
                                [triangle_coordinates],
                                0,
                                info_box_color,
                                -1,
                            )

                            cv2.rectangle(
                                freeze_img,
                                (x + int(w / 5), y + h + int(pivot_img_size / 3)),
                                (
                                    x + w - int(w / 5),
                                    y + h + pivot_img_size - int(pivot_img_size / 5),
                                ),
                                info_box_color,
                                cv2.FILLED,
                            )

                            cv2.putText(
                                freeze_img,
                                analysis_report,
                                (x + int(w / 3.5), y + h + int(pivot_img_size / 1.5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 111, 255),
                                2,
                            )

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
