# Pick-and-Place

The pick-and-place is performed based on the detection of the [YOLO model](https://github.com/hmhauter/lab-perception-and-maipulation/tree/object_detection/object-detection). 
After the 96-well plate is segmented by the YOlO mode, the grasp point is estimated. For estimation of the grasp point the plate is reconstructed as a point cloud and either PCA or registration is used to pick-up the plate. Then, the plate is transfered to tha place location marked with an ArUco marker. To not spill any liquid present in the wells, the force sensor of the robot arm is used for gentle placing. 

Additionally, the rotation of the plate is determined through Optical Character Recognition (OCR). Thereby, different challenges need to be overcome: The characters are engraved, embossed or printed on opaque, translucent or transparent plates. That makes the appearance of the characters diverse. Additionaly, the characters are not always visible to the camera.  
Three detection methods for OCR are tested: Existing models (tesseract, easyOCR), a self-trained CNN and template matching.
To ensure visibility of the characters to the camera, three modes are tested: Stationary light, sampling of a grid for camera poses, Next-Best-View.

The system is implemented in ROS Noetic for Ubuntu 20.04. 

## Installation
### Hardware 
First the harware components need to be set-up. The project uses a UR5e, a Festo Ring Light SBAL-C6-R-W-D connected with a MOXA box, the Robotiq Hand-E gripper and an Intel RealSense Camera D415. Since ROS is used it should be easy to exchange the hardware and their corresponding nodes. 

## Software
Start by installing ROS Noetic as described [here](http://wiki.ros.org/noetic/Installation/Ubuntu). Also, set-up the catkin workspace as described in the installation steps.
Additionally, set-up the virtual environment:

```python
python -m venv pick-and-place-venv
```
```python
source pick-and-place-venv/bin/activate
```
```python
pip install -r requirements.txt
```

Navigate to the ROS folder:
```console
cd ~/catkin_ws/src
```

Next, install the [Universal Robot ROS Driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver) by following the described steps or by running:
```console
sudo apt install ros-noetic-ur-robot-driver
```

Next, install the [ROS - Industrial Universal Robot package with MoveIt configurations](https://github.com/ros-industrial/universal_robot) by following the described steps or by running:
```console
sudo apt-get install ros-noetic-universal-robots
```

Next, install the [ROS package for the Intel RealSense Camera](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy?tab=readme-ov-file) by following the described steps or by running:
```console
sudo apt-get install ros-noetic-realsense2-camera
```

Next, install the unofficial [ROS Robotiq package](https://github.com/cambel/robotiq) from the Control Robotics Intelligence Group from the Nanyang Technological University, Singapore by following the described steps. The package is used for visualization but not for connecting to the Hand-E.

Next, make sure that the `ur_master_thesis` package is inside your `~/catkin_ws/src` workspace together with all the ROS packages that just were installed. 

Install any missing dependencies using rosdep:
```console
rosdep update
rosdep install --from-paths . --ignore-src -y
```

Then compile your ROS workspace (I choose to use):
```console
cd ~/catkin_ws && catkin_make
```

Open a console and start the ROS Master:
```console
roscore
```
if not already set as default do not forget to source the correct workspace:
```console
source ~/catkin_ws/devel/setup.bash
```

Please note that you have to exchange the calibration file `ur5e_umh_calibration.yaml` since it depends on the hardware you are using by running 
```console
roslaunch ur_calibration calibration_correction.launch \
  robot_ip:=<robot_ip> target_filename:="${HOME}/my_robot_calibration.yaml"
```
## Results

The video summarizes the process of pick-and-placing the plate while determining the correct rotation with a stationary, external light source:

https://github.com/user-attachments/assets/716d31c7-090e-4c12-b9dc-916908a00df8

