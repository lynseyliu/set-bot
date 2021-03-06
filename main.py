import numpy as np
import cv2
import math
import os
import argparse
from skimage.measure import compare_ssim

import pyttsx3
engine = pyttsx3.init()

# voice_id = "com.apple.speech.synthesis.voice.karen"
voice_id = "com.apple.speech.synthesis.voice.daniel"
# voice_id = "com.apple.speech.synthesis.voice.Alex"
engine.setProperty('voice', voice_id)

import process_frame
import play_set

parser = argparse.ArgumentParser(description='Play set!')
parser.add_argument('-play', action='store_true', help='play set against me')
parser.add_argument('-check', action='store_true', help='check board for sets')
args = parser.parse_args()

cap = cv2.VideoCapture(0)

i = 0
prev_frame = cv2.imread('img/board-med.png')
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Process every 15th frame
    if i % 15 == 0:
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (sim_score, diff) = compare_ssim(gray_prev, gray_curr, full=True) # sim_score 1 means exact match
        prev_frame = frame

        cards = []
        process_frame.classify(frame, cards)
        if len(cards) >= 12 and sim_score < 0.999:
            set_cards, indices = play_set.find_sets(cards)
            if args.play:
                if len(set_cards) != 0:
                    engine.say("Set!")
                    for card in set_cards:
                        engine.say(str(card))
                    engine.runAndWait()
            elif args.check:
                if len(set_cards) == 0:
                    engine.say("No sets, lay down three more!")
                    engine.runAndWait()

    # Display resulting frame
    cv2.imshow('set-bot', frame)
    cv2.waitKey(25)

    i += 1

cap.release()
cv2.destroyAllWindows()
