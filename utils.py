from model.architecture import *
import cv2

def read_image(image_path:str):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def initialize_model():
    face_encoder = InceptionResNetV2()
    path = "model/facenet_keras_weights.h5"
    face_encoder.load_weights(path)
    return face_encoder

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std