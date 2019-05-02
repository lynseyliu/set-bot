import numpy as np
import cv2
import math
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets
from torchvision import transforms, utils

import play_set

# Define model
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*23*23, 512)
        self.fc2 = nn.Linear(512, 81)

    def forward(self, x):
        # Define the forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        # print(x.shape)
        x = x.view(-1, 128*23*23)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def loss(self, prediction, label, reduction='elementwise_mean'):
        loss_val = F.cross_entropy(prediction, torch.tensor(np.array(list(label))), reduction=reduction)
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        torch.save(self, file_path)

model = torch.load('models/010.pt')

label_to_num = dict()
nums = ['2', '3']
colors = ['red', 'green', 'purple']
fills = ['solid', 'empty', 'striped']
shapes = ['diamond', 'oval', 'squiggle']
label_num = 0
for c in colors:
    for f in fills:
        for s in shapes:
            for n in nums:
                label_str = n + '-' + c + '-' + f + '-' + s + 's'
                label_to_num[label_str] = label_num
                label_num += 1

            # Special case 1- labels
            label_str = '1-' + c + '-' + f + '-' + s
            label_to_num[label_str] = label_num
            label_num += 1

num_to_label = {v: k for k, v in label_to_num.items()}
# print(num_to_label)

shape_dict = {'oval': 'pill',
              'ovals': 'pill', 
              'diamond': 'diamond',
              'diamonds': 'diamond',
              'squiggle': 'snake',
              'squiggles': 'snake'}

def classify(frame, cards):
    # Read and process layout image
    layout = frame
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

    # cv2.drawContours(layout, card_contours, -1, (0, 255, 0))
    # cv2.imwrite('test.jpg', layout)

    # Specify dimensions of result rectangle
    # It's square because orientation is unknown
    w = 200
    h = 200

    # Result rectangle corners, Order to match approx: top left, bottom left, bottom right, top right
    rect = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], np.float32)

    # Filter card candidates, warp and classify each card
    # j = 0
    for card in card_contours:
        # Get contour and perimeter length, check perim for faulty contours
        perim = cv2.arcLength(card, True)
        # Ignore random contours in the layout image that aren't cards
        if perim < 100:
            continue
        # Get approx corners of card
        approx = cv2.approxPolyDP(card, 0.02*perim, True)
        # Reshape approx to match rect
        approx = np.squeeze(approx).astype(np.float32)
        # Warp
        try:
            transform = cv2.getPerspectiveTransform(approx, rect)
        except:
            continue
        warp = cv2.warpPerspective(layout, transform, (w, h))

        # cv2.imwrite('img/warp' + str(j) + '.jpg', warp)
        # j += 1

        # warp = cv2.imread('data/train/1-green-empty-diamond/20170311_132957_1_0.jpg')
        # warp = cv2.resize(warp, (w, h))

        # Classify
        image_tensor = torch.from_numpy(warp.reshape((3, 200, 200))).type(torch.FloatTensor)
        image_tensor = image_tensor.unsqueeze(0)
        model.eval()
        output = model(image_tensor)
        index = output.data.numpy().argmax()
        label = num_to_label[index]

        label_parts = label.split('-')
        num_class = int(label_parts[0])
        color_class = label_parts[1]
        pattern_class = label_parts[2]
        shape_class = shape_dict[label_parts[3]]

        cards.append(play_set.Card(color_class, shape_class, pattern_class, num_class))

# cards = []
# layout = cv2.imread('img/board-med.png')
# classify(layout, cards)
# print(cards)
