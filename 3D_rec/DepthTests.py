import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import time
from PIL import Image
import io
import imageio

# it can be used to display 3d point cloud, but it can overfill memory
# from mpl_toolkits.mplot3d import Axes3D

def depth_to_pointcloud_simple(depth_map, scale=1.0):
    """
    Convert a depth map into a point cloud in simple way, similar to generating height map
    and landscape in procedural generation
    """
    h, w = depth_map.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_map * scale
    x = x - w / 2
    y = y - h / 2

    points = np.stack((x, -y, z), axis=-1)  # Inverting y for correct up-down orientation
    valid_points = points[z > 0]

    return valid_points


# for torch take a look at firs box in jupyter file
# timm version: 0.6.5
# pip install --force-reinstall  timm==0.4.12 torch==1.13.0 torchaudio==0.13.0 // better run previous line
# https://www.kaggle.com/code/amarlove/midas-image-depth-estimation
# latest version of opencv don't have GUI part, so I used pip install opencv-python==4.5.5.62
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# "DPT_Large" "MiDaS_small" "DPT_Hybrid"   "DPT_BEiT_Large"
model_type = "MiDaS_small"

midas = torch.hub.load('intel-isl/MiDaS', model_type)

midas.to(device)
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)
pcd = o3d.geometry.PointCloud()

# vis2 = o3d.visualization.Visualizer()
# vis2.create_window(width=800, height=600)
# pcd2 = o3d.geometry.PointCloud()
# adjust camera
K = np.array([[700, 0, 410], [0, 700, 350], [0, 0, 1]])
dop = True
scale_factor = 0.4


def refine_depth_with_edges(depth_map, edge_map, dilation_size=5, blend_factor=0.5):
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_edges = cv2.dilate(edge_map, kernel, iterations=1)
    # weight_map = 1.0 - dilated_edges / 255.0
    weight_map = dilated_edges / 255.0
    refined_depth = (depth_map * (1 - blend_factor)) + (depth_map * weight_map * blend_factor)
    refined_depth = cv2.GaussianBlur(refined_depth, (5, 5), 0)
    return refined_depth


# simple GUI
def update_alpha(val):
    global alpha
    alpha = max(1, val)


def update_p1(val):
    global update_param1
    update_param1 = max(1, val)


def update_p2(val):
    global update_param2
    update_param2 = max(1, val)


def update_num_frames(val):
    global update_num_f
    update_num_f = max(2, val)


def update_dilation_size(val):
    global dilation_s
    dilation_s = max(1, val)


def update_blend_factor(val):
    global blend_f
    blend_f = max(2, val)


def update_edge_blur_size(val):
    global edge_blur_s
    edge_blur_s = max(3, 2 * val + 1)


def update_depth_blur_size(val):
    global depth_blur_s
    depth_blur_s = max(3, 2 * val + 1)


def update_edge_blurbi(val):
    global edge_blurbi
    edge_blurbi = max(5, val)


alpha = 1
update_param1 = 100
update_param2 = 200
update_num_f = 5
dilation_s = 5
blend_f = 1
edge_blur_s = 5
edge_blurbi = 5
depth_blur_s = 3
frame_time = 1


def update_frame_time(val):
    global frame_time
    frame_time = max(1, val)


cv2.namedWindow('Depth Output', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Alpha', 'Depth Output', 5, 200, update_alpha)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Time/100', 'Frame', 1, 150, update_frame_time)
edge_buffer = []
prev_time = time.time()
paused = False

gif_path = "Gifs/"
capturing = False
frames_for_gif = []

cap = cv2.VideoCapture(0)
while cap.isOpened():
    # if not paused:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - prev_time >= frame_time/100:

        B, G, R = cv2.split(frame)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgbatch = transform(img).to(device)

        # ege detection - attempts to improve depth map
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, update_param1, update_param2)
        edge_buffer.append(edges)

        while len(edge_buffer) > update_num_f:
            edge_buffer.pop(0)

        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
            # here can move it to cuda
            output = prediction.cpu().numpy()

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(output, alpha=alpha / 100), cv2.COLORMAP_TURBO)
        cv2.imshow('Depth Output', depth_colormap)

        if capturing:
            combined_frame = np.hstack((frame, depth_colormap))
            frames_for_gif.append(combined_frame)

        prev_time = current_time
    if paused:
        # 3D
        point_cloud = depth_to_pointcloud_simple(output, scale_factor)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        vis.clear_geometries()
        vis.add_geometry(pcd)
        print("3D should be created")
        paused = False
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('p'):  # 'p' to pause and create point cloud
        paused = True
    elif key & 0xFF == ord('s'):  # 's' to start/stop capturing
        capturing = not capturing
        if not capturing:
            print("Stopped capturing frames.")
        else:
            print("Start capturing...")
    elif key & 0xFF == ord('g'):  # 'g' to generate GIF
        if frames_for_gif:
            print("Generating GIF...")
            gif_images = [Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)) for frm in frames_for_gif]
            gif_path = "output.gif"
            imageio.mimsave(gif_path, gif_images, format='GIF', fps=10)
            print(f"GIF saved as {gif_path}")
            frames_for_gif.clear()
        else:
            print("No frames captured for GIF.")
    elif key & 0xFF == 27:  # 'Esc' to exit
        break

    vis.poll_events()
    vis.update_renderer()

vis.destroy_window()
cap.release()
cv2.destroyAllWindows()

if device.type == 'cuda':
    torch.cuda.empty_cache()
