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

import cv2 as cv

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv.INTER_AREA):
    # Assuming this function is correctly implemented above
    pass

def face_detection(img):
    image = resize_with_aspect_ratio(img, width=900)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.1, 3)
    eyes = eye_cascade.detectMultiScale(image_gray, 1.1, 14)
    cropped_face = None  # Initialize cropped_face variable

    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in eyes:
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if len(faces) == 1:  # Check if exactly one face is detected
        x, y, w, h = faces[0]  # Get the coordinates of the detected face
        cropped_face = image[y:y+h, x:x+w]  # Crop the image to the face

    return image, len(faces), cropped_face  # Return the cropped face image as well


