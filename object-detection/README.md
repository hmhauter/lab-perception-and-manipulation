# Object Detection 

A YOLO model is trained on a synthetic dataset for 96-well plate detection. The synthetic images are generated through modeling of 3D scenes in IsaacSim and capturing images of the plates in diverse environments. Parameters like light conditions, backgrounds and material of the plates are varied. 

Next, a YOLO model is trained on the [DTU HPC cluster](https://www.hpc.dtu.dk/?page_id=2129) with GPU access. The test, validation and testing scripts can be found in this repository.

## Installation
For the syntetic data generation IsaaSim has to be installed. The code from `generator.py` needs to be copy-pasted into IsaacSim and get directly executed in the IsaacSim environment.

To setup the training for YOLO execute the following assuming Python >= 8 is installed:

```python
python -m venv object-detection-venv
```
```python
source object-detection-venv/bin/activate
```
```python
pip install -r requirements.txt
```

To train the YOLO model either execute `train_yolo_segm.py` locally if you have a good GPU available. For this thesis the DTU HPC cluster is used. The execution scipt can be found in `train.sh`.

## Results
After training, the numerous confusion matrix for the YOLO model is:

<img src="https://github.com/hmhauter/lab-perception-and-maipulation/blob/object_detection/figures/confusion_matrix.png" width="700">
The evaluation was done on the real-life dataset with an IoU threshold of 70% and a confidence threshold of 70%. This means that predictions with an IoU less than 70% or a confidence less than 70% are considered False Negatives. 
The overall precision is 98.02%, and the recall 92.5%. The overall accuracy of the detection model is 90.82%.

Exaples of predictions are given like:

<img src="https://github.com/hmhauter/lab-perception-and-maipulation/blob/object_detection/figures/GoodScore.png" width="700">