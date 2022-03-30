# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:20:54 2022

@author: havci
"""

import cv2 as cv
import os
from PIL import Image

cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")
cascadePath = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
cascade = cv.CascadeClassifier(cascadePath)

def detect_face(gray_frame):
    faces = cascade.detectMultiScale(gray_frame, minNeighbors=5, scaleFactor=1.5)
    if faces is None:
        return None
    for (x, y, h, w) in faces:
        return gray_frame[y:y + h, x:x + w]

def save_photo(saved_Image, person_name, image_name):
    pathName = os.path.join(image_dir, person_name)
    if os.path.exists(pathName):
        print("Directory ", pathName, " already exists")
    else:
        os.mkdir(pathName)
        print("Directory ", pathName, " Created ")
    image = Image.fromarray(saved_Image)
    name = pathName + "/" + str(image_name) + '.jpg'
    image.save(name)


def face_saved(capture, folder_name):
    if folder_name is None:
        folder_name = "default"
    img_count = 0
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break
        cropped_image = detect_face(frame)
        if cropped_image is not None:
            frame = cv.resize(cropped_image, (200, 200))
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            save_photo(gray_frame, folder_name, img_count)
            cv.putText(gray_frame, str(img_count), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('face cropper', gray_frame)
            img_count += 1
            if cv.waitKey(20) & 0xFF == ord('q') or img_count == 100:
                break