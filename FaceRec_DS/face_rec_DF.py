from deepface import DeepFace
import cv2

models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
# so, example iof how to use with video - realtime.py impementation. insted of path use numpy array
DeepFace.stream(db_path="captured_faces/", model_name=models[3])
