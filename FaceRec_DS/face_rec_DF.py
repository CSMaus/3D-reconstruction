from deepface import DeepFace
models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
DeepFace.stream(db_path="captured_faces", model_name=models[1])

verification = DeepFace.verify(img1_path="img1.jpg", img2_path = "img2.jpg")
