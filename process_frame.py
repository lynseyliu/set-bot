import numpy as np
import cv2
import math

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

    # White balance, contrast?

    # It's feature extraction time!
    # ................................................
    # Color segmentation
    # Set color bounds, Order: red, green, purple
    color_bounds = [
        # ([0, 0, 50], [80, 80, 255]),
        # ([17, 100, 15], [50, 200, 56]),
        ([30, 0, 30], [255, 80, 255]),
    ]
    for (lower, upper) in color_bounds:
        # Convert to np arrays
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
    
        # Find pixels within bounds and apply mask
        mask = cv2.inRange(warp, lower, upper)
        output = cv2.bitwise_and(warp, warp, mask=mask)
        cv2.imwrite('color' + str(i) + '.jpg', output)

    # warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    # ret, warp_thresh = cv2.threshold(warp_gray, 127, 255, cv2.THRESH_BINARY_INV)
    # warp_thresh = 255 - warp_thresh
    # warp_thresh, shape_contours, hierarchy = cv2.findContours(warp_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(warp, shape_contours, 0, (0, 0, 255), 1, 8, hierarchy)
    # print(hierarchy)
    # print('-------')
    # cv2.imwrite('shapes' + str(i) + '.jpg', warp)
