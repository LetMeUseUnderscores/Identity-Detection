from faceDetection import face_detection
from predict import predict_race
import cv2 as cv
import time

camera = cv.VideoCapture(0)

while True:
    check, frame = camera.read()
    if not check:
        print("Failed to grab a frame")
        break

    # Predict the race
    race_prediction = predict_race(frame)

    # Add the prediction text to the frame
    cv.putText(frame, race_prediction, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the prediction
    cv.imshow('Camera', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv.destroyAllWindows()