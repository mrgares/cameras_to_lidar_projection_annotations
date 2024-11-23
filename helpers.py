import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
import os.path as osp
from PIL import Image
from typing import Tuple, List, Iterable
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from skimage.transform import resize

# Display the image with the points overlaid
def visualize_pointcloud_on_image(image, points, coloring):
    # Convert the image to a format suitable for matplotlib if necessary
    if hasattr(image, 'to_ndarray'):
        image = image.to_ndarray()
    print(type(image))
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Show the image
    ax.imshow(image)

    # Overlay the points, with coloring representing intensity or depth
    scatter = ax.scatter(points[0, :], points[1, :], c=coloring, s=1, cmap='jet', alpha=0.7)

    # Add a colorbar to show intensity or depth scale
    plt.colorbar(scatter, ax=ax, label="Intensity/Depth")
    
    # Display the plot
    plt.show()
    

# Extract 2D image points and mask for each detection in camera_sample.pseudo_masks
def assign_labels_from_masks(points_projected, camera_sample):
    label_to_int = {'human': 1, 'vehicle': 2}  
    image_width, image_height = camera_sample.metadata.width, camera_sample.metadata.height
    # Initialize labels array with a default value, e.g., -1 for "no label"
    point_classes = np.zeros(points_projected.shape[1], dtype=int)

    if not hasattr(camera_sample, 'pseudo_masks'):  # No pseudo masks available
        return point_classes

    for detection in camera_sample.pseudo_masks.detections:
        mask = detection.mask
        bbox = detection.bounding_box  # [x_min, y_min, width, height]
        label = detection.label
        label_int = label_to_int.get(label, 0)
        
        # Scale bounding box coordinates to match image dimensions
        x_min = round(bbox[0] * image_width)
        y_min = round(bbox[1] * image_height)
        width = round(bbox[2] * image_width)
        height = round(bbox[3] * image_height)

        # Clip bounding box to image bounds
        x_max = min(x_min + width, image_width)
        y_max = min(y_min + height, image_height)
        x_min = max(0, x_min)
        y_min = max(0, y_min)

        # Adjust mask dimensions to fit the bounding box region
        target_height = y_max - y_min
        target_width = x_max - x_min

        if mask.shape != (target_height, target_width):
            # Resize mask to match target dimensions
            mask = resize(mask, (target_height, target_width), preserve_range=True, order=0).astype(bool)

        # Create a full-sized mask initialized to False
        full_mask = np.zeros((image_height, image_width), dtype=bool)
        full_mask[y_min:y_max, x_min:x_max] = mask

        # Find points within the mask
        x_coords = np.floor(points_projected[0, :]).astype(int)
        y_coords = np.floor(points_projected[1, :]).astype(int)
        
        # Ensure points are within image bounds
        in_bounds = (x_coords >= 0) & (x_coords < image_width) & \
                    (y_coords >= 0) & (y_coords < image_height)
                    
        # Apply the mask to label points
        in_mask_indices = np.where(in_bounds & full_mask[y_coords, x_coords])[0]
        point_classes[in_mask_indices] = label_int 

    return point_classes


def visualize_full_mask_on_image(image, pseudo_masks):
    # Convert the image to a numpy array if necessary
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    # Get the image dimensions
    img_height, img_width = image.shape[:2]

    # Create a plot to show the image with masks
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # Iterate over each detection to overlay its mask
    for detection in pseudo_masks.detections:
        # Extract the bounding box and mask
        bbox = detection.bounding_box  # [x_min, y_min, width, height]
        mask = detection.mask  # Cropped mask within the bounding box
        
        x_min, y_min, width, height = bbox
        # scale the bounding box coordinates to the image size
        x_min *= img_width
        y_min *= img_height
        width *= img_width
        height *= img_height

        # Create a full-sized mask initialized to False
        full_mask = np.zeros((img_height, img_width), dtype=bool)

        # Place the cropped mask within the bounding box location in the full mask
        full_mask[int(y_min):int(y_min + height), int(x_min):int(x_min + width)] = mask
        # Overlay the mask on the image
        ax.imshow(full_mask, cmap="jet", alpha=0.5)
    
    print()

    plt.axis("off")
    plt.show()


def map_pointcloud_to_image_with_indices(nusc: NuScenes,
                                         pointsensor_token: str,
                                         camera_token: str,
                                         min_dist: float = 1.0,
                                         render_intensity: bool = False,
                                         show_lidarseg: bool = False,
                                         filter_lidarseg_labels: List = None,
                                         lidarseg_preds_bin_path: str = None,
                                         show_panoptic: bool = False) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane. Additionally returns indices of the visible points in the original point cloud.
    """

    # Original code for retrieving camera and point sensor data
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    pc = LidarPointCloud.from_file(pcl_path) if pointsensor['sensor_modality'] == 'lidar' else RadarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    # Transformations (as per the original function)
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Get the depths (z-coordinate in the camera frame)
    depths = pc.points[2, :]

    # Prepare coloring
    if render_intensity:
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities
    elif show_lidarseg or show_panoptic:
        # lidarseg/panoptic coloring logic
        coloring = depths
    else:
        coloring = depths

    # Project points to the image plane
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Create a mask for points within the image bounds and above min_dist
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

    # Filter points and coloring based on the mask
    points = points[:, mask]
    coloring = coloring[mask]

    # NEW: Get the indices of visible points
    visible_indices = np.where(mask)[0]

    return points, coloring, im, visible_indices

def visualize_pointcloud_with_labels(pcd, point_classes):
    # Visualize the point cloud with labels

    # Step 1: Define color mappings for each class label
    color_map = {
        0: [0.5, 0.5, 0.5],  # Gray for unlabeled points
        1: [0, 1, 1],        # Cyan for "human"
        2: [1, 0, 0],        # Red for "vehicle"
    }

    # Step 2: Convert `point_classes` to a color array
    point_colors = np.array([color_map[label] for label in point_classes])

    # Step 3: Assign the color array to the existing `pcd` object
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # Step 4: Visualize the colored point cloud
    o3d.visualization.draw_geometries([pcd])
    
    