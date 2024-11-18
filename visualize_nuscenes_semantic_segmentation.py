import open3d as o3d
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os

DATASET_ROOT = "/datastore/nuScenes/"


# Initialize NuScenes dataset
nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_ROOT, verbose=False)


# Retrieve the first scene and sample
first_scene = nusc.scene[0]
first_sample_token = first_scene['first_sample_token']
first_sample = nusc.get('sample', first_sample_token)

# Get the LIDAR_TOP data token for the first sample
lidar_token = first_sample['data']['LIDAR_TOP']
lidar_data = nusc.get('sample_data', lidar_token)

# File paths for the point cloud and segmentation labels
lidar_filepath = os.path.join(nusc.dataroot, lidar_data['filename'])
lidarseg_filepath = os.path.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])

# Load the LiDAR point cloud
pc = LidarPointCloud.from_file(lidar_filepath)
points = pc.points[:3, :].T  # Use only x, y, z

# Load the segmentation labels
lidarseg_labels = np.fromfile(lidarseg_filepath, dtype=np.uint8)

# Define a colormap for the labels (example colors; modify as needed for your dataset)
# Define label groups
vehicle_labels = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
human_labels = [2, 3, 4, 5, 6, 7, 8]

# Create the label_colors dictionary
label_colors = {}

# Assign blue color to vehicle labels
for label in vehicle_labels:
    label_colors[label] = [0, 0, 1]  # Blue

# Assign red color to human labels
for label in human_labels:
    label_colors[label] = [1, 0, 0]  # Red

# Any other label defaults to gray
default_color = [0.5, 0.5, 0.5]  # Gray

# Map labels to colors using the label_colors dictionary
colors = np.array([label_colors.get(label, default_color) for label in lidarseg_labels])

# Create an Open3D point cloud object and assign points and colors
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud with labels
o3d.visualization.draw_geometries([pcd], window_name="Labeled Point Cloud")