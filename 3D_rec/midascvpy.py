import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# it can be used to display 3d point cloud, but it can overfill memory
# from mpl_toolkits.mplot3d import Axes3D


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
# "DPT_Large" "MiDaS_small" "DPT_Hybrid"
model_type = "MiDaS_small"

midas = torch.hub.load('intel-isl/MiDaS', model_type)

midas.to(device)
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

vis = o3d.visualization.Visualizer()
vis.create_window(width=1200, height=900)
pcd = o3d.geometry.PointCloud()
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


update_param1 = 100
update_param2 = 200
update_num_f = 5
dilation_s = 5
blend_f = 1
edge_blur_s = 5
edge_blurbi = 5
depth_blur_s = 3


cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Edge p1', 'Edge Detection', 10, 500, update_p1)  # 82
cv2.createTrackbar('Edge p2', 'Edge Detection', 10, 500, update_p2)  # 46
cv2.createTrackbar('Num frames', 'Edge Detection', 3, 11, update_num_frames)  # >5
cv2.namedWindow('Depth map upgraded', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Dilation s', 'Depth map upgraded', 3, 11, update_dilation_size)  # 5
cv2.createTrackbar('Blend f', 'Depth map upgraded', 1, 10, update_blend_factor)  # 0.5
cv2.createTrackbar('Blur edges', 'Edge Detection', 1, 15, update_edge_blur_size)
# cv2.createTrackbar('Blur gauss', 'Depth map upgraded', 1, 15, update_depth_blur_size)
# cv2.createTrackbar('Blur bi', 'Depth map upgraded', 5, 150, update_edge_blurbi)
edge_buffer = []

paused = False
cap = cv2.VideoCapture(0)
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

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

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(output, alpha=0.03), cv2.COLORMAP_TURBO)
        cv2.imshow('Depth Output', depth_colormap)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('p'):  # 'p' to pause and create point cloud
            paused = True
        elif key & 0xFF == 27:  # 'Esc' to exit
            break

        if len(edge_buffer) == update_num_f:
            consistent_edges = np.bitwise_and.reduce(edge_buffer)
            consistent_edges = cv2.GaussianBlur(consistent_edges, (edge_blur_s, edge_blur_s), 0)

            output2 = refine_depth_with_edges(output, consistent_edges, dilation_s, blend_f / 10)
            # output2 = cv2.normalize(output2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
            # output2 = cv2.bilateralFilter(output2, depth_blur_s, edge_blurbi, edge_blurbi)
            # output2 = output2/np.amax(output2)
            depth_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(output2, alpha=0.03), cv2.COLORMAP_TURBO)
            cv2.imshow('Edge Detection', consistent_edges)
            cv2.imshow('Depth map upgraded', depth_colormap2)

            if paused:
                # 3D
                point_cloud = depth_to_pointcloud_simple(output2, scale_factor)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud)
                vis.clear_geometries()
                vis.add_geometry(pcd)
                print("3D should be created")
                paused = False
        else:
            pass

    vis.poll_events()
    vis.update_renderer()

vis.destroy_window()
cap.release()
cv2.destroyAllWindows()

if device.type == 'cuda':
    torch.cuda.empty_cache()
