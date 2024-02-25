import open3d as o3d
import numpy as np
import cv2
import torch

vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)
pcd = o3d.geometry.PointCloud()

vis0 = o3d.visualization.Visualizer()
vis0.create_window(width=800, height=600)
pcd0 = o3d.geometry.PointCloud()

vis1 = o3d.visualization.Visualizer()
vis1.create_window(width=800, height=600)
pcd1 = o3d.geometry.PointCloud()


def calculate_frame_depthmap(img):
    """
    For current frame caculate depth map using MiDaS
    """
    imgbatch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
        return prediction.cpu().numpy()


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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device: ", device)
# "DPT_Large" "MiDaS_small" "DPT_Hybrid"   "DPT_BEiT_Large"
model_type = "MiDaS_small"
midas = torch.hub.load('intel-isl/MiDaS', model_type)
midas.to(device)
midas.eval()
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

im0 = cv2.imread("imgs/image_0.png", cv2.COLOR_BGR2RGB)
im1 = cv2.imread("imgs/image_1.png", cv2.COLOR_BGR2RGB)

print("Max img val: ", np.amax(im0))

depth_map0 = calculate_frame_depthmap(im0)
depth_map1 = calculate_frame_depthmap(im1)

scale_factor = 0.5
point_cloud0 = depth_to_pointcloud_simple(depth_map0, scale_factor)
point_cloud1 = depth_to_pointcloud_simple(depth_map1, scale_factor)

pcd0.points = o3d.utility.Vector3dVector(point_cloud0)
vis0.clear_geometries()
vis0.add_geometry(pcd0)

pcd1.points = o3d.utility.Vector3dVector(point_cloud1)
vis1.clear_geometries()
vis1.add_geometry(pcd1)

# ICP registration
threshold = 0.02
trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd0, pcd1, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

pcd1.transform(reg_p2p.transformation)

pcd_combined = pcd0 + pcd1
vis.clear_geometries()
vis.add_geometry(pcd_combined)

# display all together
while True:
    vis0.poll_events()
    vis0.update_renderer()

    vis1.poll_events()
    vis1.update_renderer()

    vis.poll_events()
    vis.update_renderer()

    if cv2.waitKey(1) & 0xFF == 27:
        break

vis0.destroy_window()
vis1.destroy_window()
vis.destroy_window()



