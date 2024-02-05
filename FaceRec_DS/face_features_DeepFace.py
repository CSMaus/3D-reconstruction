from deepface import DeepFace
import os
import pickle

image_dir = "captured_faces/"
representations = []
names = []

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(image_dir, filename)
        img = DeepFace.detectFace(path, enforce_detection=False)
        representation = DeepFace.represent(img_path=img, model_name='VGG-Face', enforce_detection=False)

        representations.append(representation)
        names.append("Kseniia")

with open("deepface_representations.pkl", "wb") as f:
    pickle.dump({"representations": representations, "names": names}, f)
