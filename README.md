# face-recognition-tf2
Face Recognition based on FaceNet for educational purposes

## System Architecture

![FaceNet Architecture](assets/FaceNet&#32;System.png)

### Face Detector

This projetct uses [MTCNN Face Detector](https://github.com/ipazc/mtcnn/).

### FaceNet 

This project uses the FaceNet architecture implemented on Tensorflow 2 by [R4j4n](https://github.com/R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X)

### Classifier

The final classifier is a Fully Connected Network using the Keras API from Tensorflow 2.


## Setup dataset

In `dataset` folder add subfolders with the faces that you wanna detect. Each subfolder should contain images from an unique person. Example:

```
dataset/
    person_1/
        1.jpg
        2.jpg
    person_1/
        3.jpg
        4.jpg
```

## Training the classifer

In root folder, run `python train.py`. This will generate a `labelmap.txt` that will be used by `detect.py` and a `classifier` folder containing the trained model.

## Detecting faces

You can run the detector on a specific image with the following command:

```bash
python detect.py --image_path datasets/person/image.jpg
```

Otherwise, you can run the detector on your webcam with:

```bash
python detect.py
```