import numpy as np
import cv2
import math
import os

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Process frame

    # Display resulting frame
    cv2.imshow('Isabelle', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
