import cv2
import torch
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# "DPT_Large" "MiDaS_small" "DPT_Hybrid"   "DPT_BEiT_Large"
model_type = "MiDaS_small"
midas = torch.hub.load('intel-isl/MiDaS', model_type)
midas.to(device)
midas.eval()
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform


def calculate_frame_depthmap(frame):
    """
    For current frame caculate depth map using MiDaS
    """
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
        # here can move it to cuda
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


threshold = 25000  # threshold for estimate_camera_motion. need to update later


def estimate_camera_motion(depth_map1, depth_map2, frame1, frame2):
    """
    Estimate camera motion between two frames using depth maps.
    depth_map1, depth_map2: Depth maps for frame1 and frame2
    frame1, frame2: Two consecutive frames
    """
    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)

    points_3d_frame1 = []
    for y in range(flow.shape[0]):
        for x in range(flow.shape[1]):
            depth = depth_map1[y, x]
            if depth > 0:  # Check for valid depth
                points_3d_frame1.append([x * depth, y * depth, depth])

    points_3d_frame1 = np.array(points_3d_frame1)

    # Get 3D points from the second frame
    points_3d_frame2 = []
    for y in range(flow.shape[0]):
        for x in range(flow.shape[1]):
            dx, dy = flow[y, x]
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < flow.shape[1] and 0 <= new_y < flow.shape[0]:
                depth = depth_map2[int(new_y), int(new_x)]
                if depth > 0:
                    points_3d_frame2.append([new_x * depth, new_y * depth, depth])

    points_3d_frame2 = np.array(points_3d_frame2)

    # Estimate motion using ICP (Iterative Closest Point)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points_3d_frame2)
    distances, indices = nbrs.kneighbors(points_3d_frame1)

    # Filter matched points
    valid_idx = np.where(distances < threshold)  # Define a threshold
    valid_indices = valid_idx[0]
    final_indices = indices[valid_idx]

    matched_points_1 = points_3d_frame1[valid_indices]
    # matched_points_2 = points_3d_frame2[indices[valid_idx]]

    matched_points_2 = points_3d_frame2[final_indices]

    # Estimate rigid transformation (rotation and translation)
    R, t = estimate_rigid_transform(matched_points_1, matched_points_2)

    return R, t


def estimate_rigid_transform(A, B):
    """
    Estimate rigid transform from A to B.
    A, B: Nx3 matrices.
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    A -= centroid_A
    B -= centroid_B

    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B.T - R @ centroid_A.T

    return R, t


def triangulate_points_between_frames(pose1, pose2, points1, points2, depth_map1, depth_map2):
    """
    Triangulate points between two frames using depth maps and camera poses.

    pose1, pose2: The camera poses (rotation and translation) for each frame.
    points1, points2: The 2D points in each frame.
    depth_map1, depth_map2: The depth maps for each frame.
    """
    # Convert poses to transformation matrices
    T1 = pose_to_transformation_matrix(pose1)
    T2 = pose_to_transformation_matrix(pose2)

    points_3d_frame1 = []
    points_3d_frame2 = []

    # triangulate points for each frame
    for point in points1:
        x, y = point
        depth = depth_map1[y, x]
        if depth > 0:
            point_3d = np.dot(T1, np.array([x * depth, y * depth, depth, 1]))
            points_3d_frame1.append(point_3d[:3])

    for point in points2:
        x, y = point
        depth = depth_map2[y, x]
        if depth > 0:
            point_3d = np.dot(T2, np.array([x * depth, y * depth, depth, 1]))
            points_3d_frame2.append(point_3d[:3])

    return np.array(points_3d_frame1), np.array(points_3d_frame2)


def pose_to_transformation_matrix(pose):
    """
    Convert a pose (rotation and translation) to a 4x4 transformation matrix.
    """
    R, t = pose
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1
    return T


def get_3d_points_from_depth(depth_map, pose):
    """
    Get 3D points from a depth map using the given camera pose.
    """
    points_3d = []
    for y in range(depth_map.shape[0]):
        for x in range(depth_map.shape[1]):
            depth = depth_map[y, x]
            if depth > 0:  # Valid depth
                point_3d = np.dot(pose, np.array([x * depth, y * depth, depth, 1]))
                points_3d.append(point_3d[:3])
    return np.array(points_3d)


def transform_points_to_global(points, pose):
    """
    Transform the given points to the global coordinate system using the given pose.
    """
    transformed_points = []
    for point in points:
        # convert to homogeneous coordinates and transform
        transformed_point = np.dot(pose, np.append(point, 1))[:3]
        transformed_points.append(transformed_point)
    return np.array(transformed_points)


cap = cv2.VideoCapture('test_video.mp4')
ret, prev_frame = cap.read()
prev_depth_map = calculate_frame_depthmap(prev_frame)

cumulative_pose = np.eye(4)

global_point_cloud = o3d.geometry.PointCloud()  # Using Open3D PointCloud
import time
print_time = True
frame_idx = 0
while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % 4 == 0:
        if print_time:
            start = time.time()
        curr_depth_map = calculate_frame_depthmap(curr_frame)
        R, t = estimate_camera_motion(prev_depth_map, curr_depth_map, prev_frame, curr_frame)

        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = t.squeeze()
        cumulative_pose = cumulative_pose @ new_pose

        points_3D = get_3d_points_from_depth(curr_depth_map, cumulative_pose)

        # Convert to Open3D point cloud and add to global point cloud
        current_point_cloud = o3d.geometry.PointCloud()
        current_point_cloud.points = o3d.utility.Vector3dVector(points_3D)
        global_point_cloud += current_point_cloud

        prev_frame = curr_frame
        prev_depth_map = curr_depth_map
        prev_pose = cumulative_pose
        if print_time:
            print("Time for one frame: ", time.time() - start)
            print_time = False

    if frame_idx % 10 == 0:
        print(frame_idx)
    if frame_idx == 320:
        print("should be ended")

cap.release()
print("released. drawing point cloud")
o3d.visualization.draw_geometries([global_point_cloud])





