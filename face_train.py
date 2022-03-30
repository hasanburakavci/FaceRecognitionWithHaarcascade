import os
import numpy as np
from PIL import Image
import cv2 as cv
import pickle

cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
cascadePath = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
cascade = cv.CascadeClassifier(cascadePath)
model = cv.face.LBPHFaceRecognizer_create()


a = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(a)
image_dir = os.path.join(BASE_DIR, "images")
x_train = []
y_labels = []
label_ids = {}

def train_method():
    os.remove("trainner.yml")
    print("Run trainer")
    current_id = 0
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                pil_image = Image.open(path).convert('L')
                image_array = np.array(pil_image, 'uint8')
                faces = cascade.detectMultiScale(image_array)
                for x, y, w, h in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)
    with open("labels_pickle", "wb") as f:
        pickle.dump(label_ids, f)

    with open("pickle_label", "wb") as f:
        pickle.dump(label_ids, f)

    model.train(x_train, np.array(y_labels))
    model.save("trainner.yml")

