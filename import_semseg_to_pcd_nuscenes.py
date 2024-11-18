from nuscenes.nuscenes import NuScenes
import open3d as o3d
import numpy as np
import os
from nuscenes.lidarseg.lidarseg_utils import paint_points_label
from nuscenes.utils.data_classes import LidarPointCloud
import fiftyone as fo

DATASET_ROOT = "/datastore/nuScenes/"
vehicle_labels = [17, 23, 16, 15, 14, 21, 18, 22, 19, 20, 31]
human_labels = [2, 3, 4, 5, 6, 7, 8]
filter_lidarseg_labels = vehicle_labels + human_labels

nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_ROOT, verbose=False)
dataset = fo.load_dataset("nuscenes")




def load_lidar(lidar_token):

    #Grab and Generate Colormaps
    lidarseg_filepath = os.path.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])


    name2index = nusc.lidarseg_name2idx_mapping

    coloring = paint_points_label(lidarseg_filepath, 
                                    filter_lidarseg_labels, 
                                    name2index, 
                                    colormap=nusc.colormap)

    lidar_filepath = nusc.dataroot + nusc.get("sample_data", lidar_token)['filename']
    root, extension = os.path.splitext(lidar_filepath)

    #Load Point Cloud
    cloud = LidarPointCloud.from_file(lidar_filepath)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points[:3,:].T)
    colors = coloring[:,:3]
    colors.max()
    pcd.colors = o3d.utility.Vector3dVector(colors)

    #Visualize Point Cloud
    o3d.visualization.draw_geometries([pcd], window_name="Labeled Point Cloud")

    #Save back Point Cloud
    o3d.io.write_point_cloud(root, pcd)
    print("Colored Point Cloud Saved at: ", root)
    #Return Filepath For New Point Cloud   
    return root



lidar_token = dataset.select_group_slices(["LIDAR_TOP"]).first().sample_token
load_lidar(lidar_token)