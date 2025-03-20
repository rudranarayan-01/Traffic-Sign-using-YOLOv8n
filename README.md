

# Traffic Sign Recognition using YOLOv8n on Google Colab

This project implements a Traffic Sign Recognition system using the YOLOv8n (Nano) model. The dataset for training is sourced from Roboflow, and Google Colab is used for high-performance training and evaluation.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Training the YOLOv8n Model](#training-the-yolov8n-model)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Prerequisites
Ensure you have the following:
- Google Account for accessing Colab
- Basic understanding of Python and Machine Learning
- Dataset from Roboflow

## Project Structure
```bash
Traffic-Sign-Recognition-YOLOv8n/
├── dataset/                    # Downloaded from Roboflow
├── yolov8_traffic_sign.ipynb   # Colab notebook for training and evaluation
├── results/                    # Model checkpoints and output results
└── README.md
```

## Setup and Installation
1. Open Google Colab at [https://colab.research.google.com](https://colab.research.google.com).
2. Upload the `yolov8_traffic_sign.ipynb` notebook.
3. Ensure you have the following dependencies in your Colab environment:
    ```bash
    !pip install ultralytics roboflow
    ```

## Data Preparation
1. Go to [Roboflow](dataset: https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou) and select a traffic sign dataset.
2. Generate the dataset in YOLO format.
3. Copy the API link to download the dataset in your Colab notebook using:
    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace().project("PROJECT_NAME")
    dataset = project.version(VERSION_NUMBER).download("yolov8")
    ```

## Training the YOLOv8n Model
1. Import Ultralytics and train the model:
    ```python
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')

    model.train(data='path_to_data.yaml', epochs=10, batch=16)
    ```

    or

   ```
    !yolo task=detect mode=train model=yolov8n.pt data=/content/RoadSignDetection/Self-Driving-Cars-6/data.yaml epochs=10 imgsz=640 batch=16
   ```
3. Adjust the number of epochs and batch size based on your Colab GPU availability.

## Evaluation
Evaluate the model using the validation set:
```python
model.val(data='path_to_data.yaml')
```

## Inference
Perform inference on images using the trained model:
```python
results = model.predict(source='path_to_test_image.jpg', save=True)
```

## Results
- Model performance will be displayed with metrics like mAP, Precision, and Recall.
- Results will be saved in the `runs/detect` folder.

## Acknowledgments
- Dataset from [Roboflow](https://www.roboflow.com/)
- YOLOv8 by Ultralytics
- Google Colab for free GPU access


