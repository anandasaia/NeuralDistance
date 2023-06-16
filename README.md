# NeuralDistance
### Leveraging neural networks for high-precision object detection and realtime distance measurements
## Realtime Object Detection and Distance Estimation

This project utilizes the YOLOv3 (You Only Look Once) object detection algorithm to detect objects in images or videos and estimate their distances from the camera. The project also includes a neural network model for distance estimation based on specific object annotations.

## Requirements

The project requires the following dependencies to be installed:

-   Python 3.x
-   TensorFlow 2.x
-   OpenCV
-   scikit-learn
-   matplotlib

To install the required dependencies, run the following command:

`pip install -r requirements.txt` 

## Folder Structure

The project folder structure is as follows:

```
|-- final.ipynb
|-- scaler.pkl
|-- yolov3.names
|-- yolov3.config
|-- yolov3.weights
|-- trained_model.h5
|-- test/
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
|-- results_img/
|   |-- image1_detected.jpg
|   |-- image1_distance.jpg
|   |-- image2_detected.jpg
|   |-- image2_distance.jpg
|   |-- ...
|-- videos/
|   |-- video1.mp4
|   |-- video2.mp4
|   |-- ...
```

-   `final.ipynb`: This Jupyter Notebook file contains the complete code for the project, including object detection, distance estimation, and processing of images, videos, or webcam feed.
-   `scaler.pkl`: This file is a serialized version of the Scikit-learn `StandardScaler` object used for feature scaling during training and inference.
-   `yolov3.names`: This file contains the names of the classes that the YOLOv3 model was trained to detect. Each class name is listed on a separate line.
-   `yolov3.config`: This file specifies the configuration settings for the YOLOv3 model architecture.
-   `yolov3.weights`: These pre-trained weights contain the learned parameters of the YOLOv3 model. They are necessary for performing object detection.
-   `trained_model.h5`: This file contains the trained neural network model for distance estimation. It is used during the inference process.
-   `test/`: This folder contains the input images for object detection and distance estimation.
-   `results_img/`: This folder stores the processed images with object detection bounding boxes and distance estimation annotations.
-   `videos/`: This folder contains input videos for object detection and distance estimation.

## Data Annotation Format

To train the neural network model for distance estimation, you need to provide object annotations in a CSV file (`annotations.csv`). The annotations should follow the following format:

`filename, xmin, ymin, xmax, ymax, distance
image1.jpg, 100, 50, 200, 150, 2.5
image1.jpg, 300, 200, 400, 300, 1.8
image2.jpg, 50, 100, 150, 200, 3.2
...` 

-   `filename`: The name of the image file where the object is present.
-   `xmin`: The x-coordinate of the top-left corner of the object bounding box.
-   `ymin`: The y-coordinate of the top-left corner of the object bounding box.
-   `xmax`: The x-coordinate of the bottom-right corner of the object bounding box.
-   `ymax`: The y-coordinate of the bottom-right corner of the object bounding box.
-   `distance`: The actual distance from the camera to the object.

Ensure that the annotations cover a diverse range of object sizes and distances to achieve accurate distance estimation.

## Usage

The `final.ipynb` Jupyter Notebook provides an interactive interface for running the object detection and distance estimation pipeline. Follow the instructions provided in the notebook to execute the following steps:

1.  Load the YOLOv3 model with pre-trained weights (`yolov3.weights`) and configuration (`yolov3.config`).
2.  Perform object detection on the input images or videos using the YOLOv3 model.
3.  Extract the object bounding boxes and crop the detected objects.
4.  Load the trained distance estimation model (`trained_model.h5`).
5.  Preprocess the cropped object images for distance estimation.
6.  Use the trained model to estimate the distances of the objects.
7.  Visualize the results by drawing bounding boxes and displaying the estimated distances.
8.  Save the processed images with bounding boxes and distance annotations in the `results_img/` folder.
9.  (Optional) Process videos or webcam feed by providing the corresponding paths.

Make sure to update the file paths and names according to your specific setup. You may also modify the code to suit your requirements, such as adding additional functionality or customizing the visualizations.

## Output

The project provides visual output in the form of processed images with object detection bounding boxes and distance estimation annotations. The output images are saved in the `results_img/` folder with the following naming convention:

-   `{image_name}_detected.jpg`: Processed image with bounding boxes around the detected objects.
-   `{image_name}_distance.jpg`: Processed image with bounding boxes and estimated distances displayed.

Additionally, if you process videos or webcam feed, the output will be displayed in real-time using OpenCV's video processing capabilities.

## Neural Network Architecture

The neural network architecture used for distance estimation is a custom model designed specifically for this project. It consists of 3 hidden layers and 1 output layer The architecture is trained on the annotated dataset to learn the relationship between object features and their corresponding distances. The trained model (`trained_model.h5`) is used for inference during the distance estimation step.

## References

If you need more information about YOLOv3, distance estimation, or the neural network architecture used in this project, please refer to the following resources:

-   [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
-   [OpenCV Documentation](https://docs.opencv.org/)
-   [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
-   [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

----------

If you have any further questions or need assistance with any specific part of the code or project, feel free to  drop in a message or open an issue. 
