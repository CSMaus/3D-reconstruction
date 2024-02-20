import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def depth_to_pointcloud(depth_map, K, scale=1.0):
    """
    Convert a depth map into a point cloud with one point for each
    pixel in the image, using the camera intrinsic matrix `K`.
    """
    z = depth_map.reshape(-1) * scale
    h, w = depth_map.shape

    x, y = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_map.reshape(-1)
    x = x.reshape(-1)
    y = y.reshape(-1)

    x3 = (x - K[0, 2]) * z / K[0, 0]
    y3 = (y - K[1, 2]) * z / K[1, 1]
    z3 = z
    points = np.stack((x3, y3, z3), axis=1)
    valid_points = points[z > 0]

    return valid_points


# for torch take a look at firs box in jupyter file
# timm version: 0.6.5
# pip install --force-reinstall  timm==0.4.12 torch==1.13.0 torchaudio==0.13.0 // better run previous line
# https://www.kaggle.com/code/amarlove/midas-image-depth-estimation
# latest version of opencv don't have GUI part, so I used pip install opencv-python==4.5.5.62
device = torch.device("cpu")  # cuda") if torch.cuda.is_available() else torch.device("cpu")
# "DPT_Large" "MiDaS_small" "DPT_Hybrid"
model_type = "MiDaS_small"

midas = torch.hub.load('intel-isl/MiDaS', model_type)

midas.to(device)
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# adjust camera
K = np.array([[700, 0, 410], [0, 700, 350], [0, 0, 1]])
dop = True

scale_factor = 1
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

    # 3d
    point_cloud = depth_to_pointcloud(output, K, scale_factor)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])

    # if cv2.waitKey(15) & 0xFF == ord('q'):
    # Esc to break the loop
    if cv2.waitKey(5) & 0xFF == 27:
        break
    break

cap.release()
cv2.destroyAllWindows()

if device.type == 'cuda':
    torch.cuda.empty_cache()
