import open3d as o3d
import fiftyone as fo

# filepath = "/datastore/nuScenes/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd"
# create new fiftyone dataset
# dataset = fo.Dataset(name="test_dataset", overwrite=True)
dataset = fo.load_dataset("test_dataset")

filepath = "/datastore/nuScenes/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883531450214.pcd"
# pcd = o3d.io.read_point_cloud(filepath)
# print colors from pcd file
# print(pcd.colors)
# o3d.visualization.draw_geometries([pcd], window_name="LIDAR_TOP")


# def load_lidar_sample(file_path):
#     sample = fo.Sample(filepath=file_path)
#     dataset.add_sample(sample)

def load_lidar_sample(file_path):
    sample = dataset['6733fafab4c4a9a677b8817f']
    sample['LIDAR_2_path'] = fo.Sample(filepath=file_path)
    
load_lidar_sample(filepath)