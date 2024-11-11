# cameras_to_lidar_projection_annotations
In this project we use instance segmentation masks to project into Lidar point clouds and generate 3D bounding boxes for a 3D Object detection training

## Setup
1. Clone the repo
2. Go to the repo directory
```bash
cd cameras_to_lidar_projection_annotations
```
3. Create a docker image with the following command:
```bash
docker build -t cuboids-generation .
```
4. Run the docker container with the following command:
```bash
docker run --name cuboids_gen -it --gpus all -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -v /path/to/datastore:/path/to/datastore -v `pwd`:/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg --shm-size=16g --network fiftyone_network -e FIFTYONE_DATABASE_URI=mongodb://fiftyone_server:27017 cuboids-generation 
```

You only add the ``-e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg`` if youw want support for OpenGL visualization in the container and using WSLg for WSL2.