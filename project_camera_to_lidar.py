from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import fiftyone as fo
import matplotlib.pyplot as plt
import helpers
import numpy as np
import open3d as o3d
import os


DATASET_ROOT = "/datastore/nuScenes/"

dataset = fo.load_dataset("nuscenes")
nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_ROOT, verbose=False)
explorer = NuScenesExplorer(nusc)

#['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'LIDAR_TOP', 

for group in dataset.iter_groups():
    lidar_sample = group["LIDAR_TOP"]
    front_camera_sample = group["CAM_FRONT"]
    front_left_camera_sample = group["CAM_FRONT_LEFT"]
    front_right_camera_sample = group["CAM_FRONT_RIGHT"]
    back_camera_sample = group["CAM_BACK"]
    back_left_camera_sample = group["CAM_BACK_LEFT"]
    back_right_camera_sample = group["CAM_BACK_RIGHT"]
    camera_sample_list = [front_camera_sample,
                        front_left_camera_sample,
                        front_right_camera_sample,
                        back_camera_sample,
                        back_left_camera_sample,
                        back_right_camera_sample]

    lidar_token = lidar_sample.sample_token
    front_camera_token = front_camera_sample.sample_token
    front_left_camera_token = front_left_camera_sample.sample_token
    front_right_camera_token = front_right_camera_sample.sample_token
    back_camera_token = back_camera_sample.sample_token
    back_left_camera_token = back_left_camera_sample.sample_token
    back_right_camera_token = back_right_camera_sample.sample_token
    camera_token_list = [front_camera_token, 
                        front_left_camera_token, 
                        front_right_camera_token,
                        back_camera_token,
                        back_left_camera_token,
                        back_right_camera_token]
    
    lidarseg_filepath = os.path.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])

    pcd = o3d.io.read_point_cloud(lidar_sample.filepath)
    lidar_data = np.asarray(pcd.points).T
    point_classes = -1 * np.ones(lidar_data.shape[1], dtype=int)
    for i, camera_token in enumerate(camera_token_list):
        points_projected, _, _, visible_indices = helpers.map_pointcloud_to_image_with_indices(nusc, 
                                                                                                    lidar_token, 
                                                                                                    camera_token)
        visible_labels = helpers.assign_labels_from_masks(points_projected, camera_sample_list[i])

        point_classes[visible_indices] = visible_labels
    break
helpers.visualize_pointcloud_with_labels(pcd, point_classes)
