import cv2
import mediapipe as mp
import numpy as np
from deepface.modules import modeling
from deepface.extendedmodels import Gender, Race, Emotion

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

emotion_model = modeling.build_model("Emotion")
age_model = modeling.build_model("Age")
gender_model = modeling.build_model("Gender")
race_model = modeling.build_model("Race")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))

            input_face = np.reshape(face, (1, 224, 224, 3))
            # print(np.shape(face))
            emotion_predictions = emotion_model.predict(input_face)
            dominant_emotion = Emotion.labels[np.argmax(emotion_predictions)]

            apparent_age = int(age_model.predict(input_face))

            gender_predictions = gender_model.predict(input_face)
            dominant_gender = Gender.labels[np.argmax(gender_predictions)]

            race_predictions = race_model.predict(input_face)
            dominant_race = Race.labels[np.argmax(race_predictions)]

            text = f"Emotion: {dominant_emotion}\nAge: {apparent_age}\nGender: {dominant_gender}\nRace: {dominant_race}"
            y0, dy = y - 30, 18
            for i, line in enumerate(text.split('\n')):
                y = y0 - i * dy
                cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Frame', img)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
face_detection.close()
