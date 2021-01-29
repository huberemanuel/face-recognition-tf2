import tensorflow as tf
import numpy as np
import cv2
import argparse

from train import detect_faces, encode_face
from utils import read_image, initialize_model


def predict(image, face_model, classifier, classes, min_score=0.8):

    faces, boxes = detect_faces(image, return_boxes=True)
    dets = []
    for face in faces:
        encoded_face = encode_face(face, face_model)
        encoded_face = np.expand_dims(encoded_face, axis=0)
        input_data = tf.constant(encoded_face)
        result = classifier(input_data).numpy()
        idx = result.argmax()
        if result.max() >= min_score:
            print(f"A wild {classes[idx]} was found!")
            dets.append(classes[idx])
        else:
            dets.append(-1)

    return dets, boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help="Image path to be analyzed")
    args = parser.parse_args()

    face_model = initialize_model()
    labelmap_path = "labelmap.txt"
    classes = {}
    classifier = tf.keras.models.load_model("classifier/")
    
    with open(labelmap_path, "r") as file:
        for i, label in enumerate(file.read().split("\n")):
            classes[i] = label

    if args.image_path:
        image = read_image(args.image_path)
        predict(image, face_model, classifier, classes)
    else:
        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            labels, boxes = predict(frame, face_model, classifier, classes)
            for box, label in zip(boxes, labels):
                if label == -1:
                    continue
                x, y, width, height = box
                cv2.rectangle(frame, (x,y), (x+width,y+height), (255,255,255), 2)
                cv2.putText(frame, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()