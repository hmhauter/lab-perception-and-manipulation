# Object Detection 

A YOLO model is trained on a synthetic dataset for 96-well plate detection. The synthetic images are generated through modeling of 3D scenes in IsaacSim and capturing images of the plates in diverse environments. Parameters like light conditions, backgrounds and material of the plates are varied. 

Next, a YOLO model is trained on the DTU HPC cluster. The test, validation and testing scripts can be found in thie repository.

## Installation
For the syntetic data generation IsaaSim has to be installed. The code from `generator.py` needs to be copy-pasted into IsaacSim and get directly executed in the IsaacSim environment.

To setup the training for YOLO execute the following assuming Python >= 8 is installed:

```python
python -m venv object-detection-venv
```
```python
source object-detection-venv/bin/activate
```

To train the YOLO model either execute `train_yolo_segm.py` locally if you have a good GPU available. For this thesis the DTU HPC cluster is used. The execution scipt can be found in `train.sh`.
