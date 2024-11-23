from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import fiftyone as fo
import matplotlib.pyplot as plt
import helpers
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm


dataset = fo.load_dataset("nuscenes")
lidar_sample = dataset.select_group_slices(['LIDAR_TOP']).match({"split": "train"}).take(1).first()

pcd = o3d.io.read_point_cloud(lidar_sample.filepath)
lidar_data = np.asarray(pcd.points).T
point_classes = np.zeros(lidar_data.shape[1], dtype=int)

helpers.visualize_pointcloud_with_labels(pcd, lidar_sample['pseudo_label'])