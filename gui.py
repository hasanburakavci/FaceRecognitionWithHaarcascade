from tkinter import *
import pickle

import cv2 as cv
from PIL import Image, ImageTk

import face_saved as fc
import face_train as ft
import os

ft.train_method()

model = cv.face.LBPHFaceRecognizer_create()
model.read('trainner.yml')
cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
cascadePath = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
cascade = cv.CascadeClassifier(cascadePath)


#pickle
pickle_labels = {"person_name": 1}
with open("pickle_label", 'rb') as f:
    pickle_labels = pickle.load(f)
    pickle_labels = {v: k for k, v in pickle_labels.items()}

def save_photo():
    fc.face_saved(capture, entry.get())

def face_train():
    ft.train_method()
    with open("pickle_label", 'rb') as f:
        pickle_labels = pickle.load(f)
        pickle_labels = {v: k for k, v in pickle_labels.items()}

photo = None
master = Tk()
master.resizable(False, False)
label = Label(master, text="Please Entry Name", width=45)
label.grid(row=0, column=0)
entry = Entry(master)
entry.grid(row=0, column=1)
label_video = Label(master)
btn_save_photo = Button(master, command=save_photo, text="Save Photo")
btn_save_photo.grid(row=1, column=0)
btn_face_train = Button(master, command=face_train, text="Manuel Face Train")
btn_face_train.grid(row=1, column=2)
label_video.grid(row=2, columnspan=2)

capture = cv.VideoCapture(0)
def update():
    isTrue, frame = capture.read()
    if isTrue:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rectangleList = cascade.detectMultiScale(gray_frame, minNeighbors=5, scaleFactor=1.5)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        for (x, y, w, h) in rectangleList:
            detect_frame = gray_frame[y:y + h, x:x + w]
            detect_frame = cv.resize(detect_frame, (200, 200))
            id_, confidence = model.predict(detect_frame)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if 45 < confidence < 85:
                cv.putText(frame, pickle_labels[id_], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.image = imgtk
        label_video.configure(image=imgtk)
    else:
        print("camera is not avaiable")
        master.quit()
    master.after(20, update)


update()
master.mainloop()
