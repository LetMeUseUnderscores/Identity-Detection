import cv2 as cv
import math

face_cascade = cv.CascadeClassifier('HaarCascadeTraining/haarcascade_frontalface_alt.xml')
eye_cascade = cv.CascadeClassifier('HaarCascadeTraining/haarcascade_eye.xml')

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

def face_detection(img):
    image = resize_with_aspect_ratio(img, width=900)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.1, 3)
    eyes = eye_cascade.detectMultiScale(image_gray, 1.1, 15)

    for(x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in eyes:
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    return image

camera = cv.VideoCapture(0)

while True:
    check, frame = camera.read()
    cv.imshow('Camera', face_detection(frame))
    key = cv.waitKey(1)
    if(key == ord('q')):
        break