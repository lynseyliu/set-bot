import numpy as np
import cv2
import math
# import matplotlib.pyplot as plt

# Read and process layout image
layout = cv2.imread('layout.jpg')
# Increase saturation
layout_hsv = cv2.cvtColor(layout, cv2.COLOR_BGR2HSV).astype('float32')
(h, s, v) = cv2.split(layout_hsv)
s = s * 2
s = np.clip(s, 0, 255)
layout_hsv = cv2.merge([h, s, v])
layout = cv2.cvtColor(layout_hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
cv2.imwrite('saturation.jpg', layout)
# Grayscale
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
classed_cards = set()
for i in range(0, len(card_contours)):
    # Get contour and perimeter length, check perim for faulty contours
    card = card_contours[i]
    perim = cv2.arcLength(card, True)
    # Ignore random contours in the layout image that aren't cards
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

    # Find shapes using contours (noisy result)
    warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    ret, warp_thresh = cv2.threshold(warp_gray, 127, 255, cv2.THRESH_BINARY_INV)
    warp_thresh = 255 - warp_thresh
    warp_thresh, shape_contours, hierarchy = cv2.findContours(warp_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Remove contour noise caused by patterns, edge
    filtered_shapes = []
    for j, c in enumerate(shape_contours):
        perim = cv2.arcLength(c, True)
        if perim > 100 and perim < 2 * (w + h) - 100:
            filtered_shapes.append(c)
    
    # Remove duplicate "inner" shape contours
    # Compare each pair of contours (will be in order outer, inner if applicable)
    shapes_copy = filtered_shapes.copy()
    j = 0
    while j < len(shapes_copy) - 1:
        shape_perim = cv2.arcLength(shapes_copy[j], True)
        next_perim = cv2.arcLength(shapes_copy[j+1], True)
        if next_perim < shape_perim - 10:
            filtered_shapes.remove(shapes_copy[j+1])
        j += 2
    
    # Set number class based on shape count
    num_class = len(filtered_shapes)

    # classed_cards.add({'shape': shape_class,
    #                    'number': num_class,
    #                    'pattern': pattern_class,
    #                    'color': color_class})