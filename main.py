import numpy as np
import cv2
import math
import os
import argparse

import process_frame
import play_set

parser = argparse.ArgumentParser(description='Play set!')
parser.add_argument('-play', action='store_true', help='play set against me')
parser.add_argument('-check', action='store_true', help='check board for no sets')
args = parser.parse_args()

cap = cv2.VideoCapture(0)

i = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Process every 15th frame
    if i % 15 == 0:
        cards = []
        process_frame.classify(frame, cards)
        sets, indices = play_set.find_sets(cards)
        print(sets)

    # Display resulting frame
    cv2.imshow('set-bot', frame)
    cv2.waitKey(25)

    i += 1

cap.release()
cv2.destroyAllWindows()
