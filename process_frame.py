import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# Read and process layout imagge
layout = cv2.imread('layout.jpg')
layout_gray = cv2.cvtColor(layout, cv2.COLOR_BGR2GRAY)
ret, layout_thresh = cv2.threshold(layout_gray, 127, 255, cv2.THRESH_BINARY_INV)
layout_thresh = 255 - layout_thresh

# Get card contours
layout_thresh, card_contours, hierarchy = cv2.findContours(layout_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Specify dimensions of result rectangle
# It's square right now because orientation is unknown
w = 200
h = 200

# Result rectangle corners, Order to match approx: top left, bottom left, bottom right, top right
rect = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], np.float32)

# Warp and classify each card
for i in range(0, len(card_contours)):
    # Get contour and perimeter length, check perim for faulty contours
    card = card_contours[i]
    perim = cv2.arcLength(card, True)
    if perim < 100:
        continue
    # Get approx corners of card
    approx = cv2.approxPolyDP(card, 0.02*perim, True)
    # Reshape approx to match rect
    approx = np.squeeze(approx).astype(np.float32)
    # Warp
    transform = cv2.getPerspectiveTransform(approx, rect)
    warp = cv2.warpPerspective(layout, transform, (w, h))

    # It's feature extraction time!
    #  |-|   _    *  __
    #  |-|   |  *    |/'
    #  |-|   |~*~~~o~|
    #  |-|   |  O o *|
    # /___\  |o___O__|

    # Find shapes using contours
    warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    ret, warp_thresh = cv2.threshold(warp_gray, 127, 255, cv2.THRESH_BINARY_INV)
    warp_thresh = 255 - warp_thresh
    warp_thresh, shape_contours, hierarchy = cv2.findContours(warp_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_shapes = []
    for i, c in enumerate(shape_contours):
        perim = cv2.arcLength(c, True)
        if perim > 100 and perim < 2 * (w + h) - 100:# and hierarchy[0][i][3] == 0:
        #if cv2.contourArea(c) > 50 and cv2.contourArea(c) < (w * h) / 2:
            #cv2.drawContours(warp, [c], -1, (255, 0, 0), 1, 8)
            filtered_shapes.append(c)
    cv2.drawContours(warp, filtered_shapes, -1, (255, 0, 0), 1, 8)
    cv2.imwrite('shapes' + str(i) + '.jpg', warp)
