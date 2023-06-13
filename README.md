# FeedForward-Distance

# Real-Time Object Detection and Distance Estimation

This repository contains code for real-time object detection and distance estimation using a neural network model. The code utilizes YOLOv3 for object detection and a trained neural network model for distance estimation. The model can be used to detect objects in images, videos, or through a webcam feed.

## Files

The repository contains the following files:

- `yolov3.cfg` and `yolov3.weights`: Configuration and weights files for the YOLOv3 model.
- `yolov3.names`: File containing class names for object detection.
- `final.ipynb`: Jupyter Notebook file containing the code for training the neural network model and performing object detection and distance estimation.
- `scaler.pkl`: Pickle file containing the scaler used for normalizing input data.
- `trained_model.h5`: HDF5 file containing the trained neural network model.
- `requirements.txt`: File specifying the required Python libraries and their versions.

## Setup and Installation

To run the code in this repository, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
