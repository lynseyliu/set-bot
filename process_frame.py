import numpy as np
import cv2
import math
# import matplotlib.pyplot as plt

# Get shape contours
pill_img = cv2.imread('pill.jpg')
pill_img = cv2.cvtColor(pill_img, cv2.COLOR_BGR2GRAY)
_, pill_img = cv2.threshold(pill_img, 127, 255, cv2.THRESH_BINARY)
_, PILL, _ = cv2.findContours(pill_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
PILL = PILL[0]

diamond_img = cv2.imread('diamond.jpg')
diamond_img = cv2.cvtColor(diamond_img, cv2.COLOR_BGR2GRAY)
_, diamond_img = cv2.threshold(diamond_img, 127, 255, cv2.THRESH_BINARY)
_, DIAMOND, _ = cv2.findContours(diamond_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
DIAMOND = DIAMOND[0]

snake_img = cv2.imread('snake.jpg')
snake_img = cv2.cvtColor(snake_img, cv2.COLOR_BGR2GRAY)
_, snake_img = cv2.threshold(snake_img, 127, 255, cv2.THRESH_BINARY)
_, SNAKE, _ = cv2.findContours(snake_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
SNAKE = SNAKE[0]

SHAPES = ['pill', 'diamond', 'snake']

# Read and process layout image
layout = cv2.imread('layout.jpg')
# Increase saturation
layout_hsv = cv2.cvtColor(layout, cv2.COLOR_BGR2HSV).astype('float32')
(h, s, v) = cv2.split(layout_hsv)
s = s * 2
s = np.clip(s, 0, 255)
layout_hsv = cv2.merge([h, s, v])
layout = cv2.cvtColor(layout_hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
# Grayscale
layout_gray = cv2.cvtColor(layout, cv2.COLOR_BGR2GRAY)
ret, layout_thresh = cv2.threshold(layout_gray, 127, 255, cv2.THRESH_BINARY_INV)
layout_thresh = 255 - layout_thresh

# Get card contours
layout_thresh, card_contours, hierarchy = cv2.findContours(layout_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Specify dimensions of result rectangle
# It's square right now because orientation is unknown ¯\_(ツ)_/¯
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

    # Make masks based on first shape
    fill_mask = np.zeros(warp.shape[:2], np.uint8)
    cv2.drawContours(fill_mask, [filtered_shapes[0]], 0, (255,255,255), -1)
    cv2.drawContours(fill_mask, [filtered_shapes[0]], 0, 0, 5)
    outline_mask = np.zeros(warp.shape[:2], np.uint8)
    cv2.drawContours(outline_mask, [filtered_shapes[0]], 0, (255,255,255), 15)
    outline_mask = fill_mask & outline_mask

    # Convert card to hsv
    warp_hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV).astype('float32')
    (mean_h, mean_s, mean_v, mean_a) = cv2.mean(warp_hsv, outline_mask)

    # Set color class based on hue value
    if mean_h >= 0 and mean_h < 40:
        color_class = 'red'
    elif mean_h >= 40 and mean_h < 100:
        color_class = 'green'
    elif mean_h >= 100:
        color_class = 'purple'

    # Get match values between card shape and actual shape contours
    pill_match = cv2.matchShapes(PILL, filtered_shapes[0], 1, 0.0)
    diamond_match = cv2.matchShapes(DIAMOND, filtered_shapes[0], 1, 0.0)
    snake_match = cv2.matchShapes(SNAKE, filtered_shapes[0], 1, 0.0)
    matches = [pill_match, diamond_match, snake_match]

    # Set shape class based on closest matching shape contour
    shape_class = SHAPES[matches.index(min(matches))]

    # Pattern