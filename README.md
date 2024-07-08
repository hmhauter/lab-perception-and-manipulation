# Robot Perception and Manipulation in Life Science Laboratories

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NVIDIA Flex](https://img.shields.io/badge/NVIDIA-Flex-green)](https://developer.nvidia.com/flex)
![Python](https://img.shields.io/badge/Python-3.8-blue)

## Description
This GitHub repository is part of the Master Thesis *Robot Perception and Manipulation in Life Science Laboratories* that was created in a cooperation between DTU Electro and Novo Nordisk.  
The Thesis explores the opportunities for systems that use computer vision and robotic manipulation to fulfill tasks in life science laboratories.

## Table of Contents

- [Installation](#installation)
- [Subprojects](#subprojects)
- [Background](#background)
- [Overview of the Project](#solution)
- [License](#license)
- [Credits](#credits)
- [Contact](#contact)

## Installation

Please refer to the specific folders of the project for installation details for the specific usage.

```python
git clone https://github.com/hmhauter/lab-perception-and-maipulation
```


## Subprojects

- [Object Detection](https://github.com/hmhauter/lab-perception-and-maipulation)
- [Pick-and-Place](https://github.com/hmhauter/lab-perception-and-maipulation)

## Background
To further advance laboratory automation, a bridge is required to connect the unordered human workspace with the structured world of automation equipment.
This work focuses on 96-well plates, commonly used in research labs, and presents a robotic system to transfer these plates, for example, from a laboratory workbench to a liquid handler. The developed system reduces the time and effort needed by researchers by automating the pick-and-place tasks of 96-well plates.

This study explores the integration of computer vision and robot manipulation in research laboratories using accessible hardware. The robot observes its environment with a camera, detecting plates using a YOLO model trained on synthetic data. The depth and segmented images are combined to accurately estimate the plate's pose with point cloud registration for precise grasping by the parallel gripper. Optical character recognition (OCR) is employed to determine the correct plate orientation by identifying the $A$-$1$ well, addressing challenges of visibility on transparent and translucent plates with embossed, engraved, and printed characters. The robot then places the plates on automation equipment, such as liquid handlers, without spilling any liquid. 

Results demonstrate a robust detection model capable of adapting to varying laboratory conditions. The work brings novelty to the state of the art by the high-accuracy OCR on embossed, engraved and printed characters on transparent and translucent plates, achieving a 97% success rate in correct plate rotation determination with the a simple ring light setup. Additionally, the work serves as a first investigation for future improvement in optimizing camera positions for character visibility using active vision techniques.

The project delivers an easy extendable setup based on the Robot Operating System (ROS) that is tested under real-world conditions, successfully replicating researcher tasks with an overall success rate of 87%. This proof-of-concept highlights the potential of combining computer vision and robotic manipulation in laboratory settings, facilitating future enhancements such as integrating a mobile base for automating routine tasks.

## Overview of the Project
An overview of the compnents and subsystems is summarized in the following Figure:
<img src="https://github.com/hmhauter/lab-perception-and-maipulation/figures/SystemOverview.PNG" width="700">

## License

This project is licensed under the terms of the MIT license. 

## Credits

This project was developed in collaboration with [Novo Nordisk](https://www.novonordisk.com/), a global healthcare company that is dedicated to improving the lives of people with diabetes and other chronic diseases. Novo Nordisk provided support and resources for this project, including expert guidance and access to equipment and facilities.

## Contact

For any inquiries or feedback, feel free to contact us:

- **LinkedIn:** [Helena Hauter](https://dk.linkedin.com/in/helena-mia-hauter)
- **Website:** [Novo Nordisk](https://www.novonordisk.com/)