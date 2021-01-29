from model.architecture import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer, LabelEncoder
from tensorflow.keras.models import load_model

from utils import initialize_model, read_image, normalize

def detect_faces(img, min_score=0.8, return_boxes=False):
    face_detector = mtcnn.MTCNN()

    detections = face_detector.detect_faces(img)
    faces = []
    boxes = []
    for detection in detections:
        if detection["confidence"] >= min_score:
            x, y, width, height = detection['box']
            xmax, ymax = x+width , y+height
            faces.append( img[y:ymax, x:xmax] )
            boxes.append((x, y, width, height))
    
    if return_boxes:
        return faces, boxes

    return faces

def encode_face(face, model, required_shape=(160, 160)):
    face = normalize(face)
    face = cv2.resize(face, required_shape)
    face_d = np.expand_dims(face, axis=0)
    return model.predict(face_d)[0]

def normalize_encode(encode):
    l2_normalizer = Normalizer('l2')
    return l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]


def process_dataset():

    encoding_dict = {}
    model = initialize_model()
    dataset_path = "dataset"

    for face_names in os.listdir(dataset_path):
        
        if not os.path.isdir(os.path.join(dataset_path, face_names)):
            continue
        
        encodes = []
        person_dir = os.path.join(dataset_path, face_names)

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir,image_name)

            img = read_image(image_path)

            faces = detect_faces(img)
            if len(faces) == 0:
                print(f"No face found at {image_path}")
                continue

            # Selecting the first face, since it is expected to each pic to have
            # only one face
            face = faces[0]
            
            encode = encode_face(face, model)
            encodes.append(encode)

        if len(encodes) > 0:
            encode = np.sum(encodes, axis=0 )
            encode = normalize_encode(encode)
            encoding_dict[face_names] = encode

    return encoding_dict

def train_classifier(encodings):
    num_faces = len(encodings)
    model = tf.keras.Sequential([
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(num_faces, activation="softmax"),
    ])
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer="adam",
        metrics=["accuracy"]
    )

    x_train = np.array([value for _, value in encodings.items()])
    y_train = np.array(list(encodings.keys()))
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)

    model.fit(x_train, y_train, epochs=10)
    model.save("classifier")

    with open("labelmap.txt", "w") as file:
        for class_ in le.classes_:
            file.write(class_ + "\n")

def main():
    print("Processings Dataset")
    encodings = process_dataset()
    print("Training classifier")
    train_classifier(encodings)

if __name__ == "__main__":
    main()