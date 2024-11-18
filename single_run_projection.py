from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import fiftyone as fo
import matplotlib.pyplot as plt
import helpers
import numpy as np
import open3d as o3d



DATASET_ROOT = "/datastore/nuScenes/"

dataset = fo.load_dataset("nuscenes")
nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_ROOT, verbose=False)
explorer = NuScenesExplorer(nusc)

#['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'LIDAR_TOP', 

lidar_sample = dataset.select_group_slices(["LIDAR_TOP"]).first()
camera_sample = dataset.select_group_slices(["CAM_BACK_LEFT"]).first()

lidar_token = lidar_sample.sample_token
camera_token = camera_sample.sample_token

points_projected, coloring, image, visible_indices = helpers.map_pointcloud_to_image_with_indices(nusc, lidar_token, camera_token)

pcd = o3d.io.read_point_cloud(lidar_sample.filepath)
lidar_data = np.asarray(pcd.points).T
point_classes = -1 * np.ones(lidar_data.shape[1], dtype=int)
visible_labels = helpers.assign_labels_from_masks(points_projected, camera_sample)

point_classes[visible_indices] = visible_labels

helpers.visualize_pointcloud_with_labels(pcd, point_classes)
