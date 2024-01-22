import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


# for torch take a look at firs box in jupyter file
# timm version: 0.6.5
# https://www.kaggle.com/code/amarlove/midas-image-depth-estimation
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# "DPT_Large" "MiDaS_small" "DPT_Hybrid"
model_type = "MiDaS_small"

midas = torch.hub.load('intel-isl/MiDaS', model_type)

midas.to(device)
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

dop = True
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
        output = prediction.cpu().numpy()
        # output = 255*output/np.amax(output)
        if dop:
            print(np.amin(output))
            print(np.amax(output))
            dop = False



    # COLORMAP_TURBO and _JET works quite well
    # do not use plt bcs it's overfill the memory or gives error bcs of conflicts in attempts of
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(output, alpha=0.03), cv2.COLORMAP_TURBO)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Depth Output', depth_colormap)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if device.type == 'cuda':
    torch.cuda.empty_cache()




