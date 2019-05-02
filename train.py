import numpy as np
import cv2
import math
import os
import io
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets
from torchvision import transforms, utils

import h5py


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


# Load dataset with transforms
class SetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.labels = []
        self.im_paths = []
        for root, dirs, files in os.walk(self.root_dir, topdown=True):
            for name in files:
                temp = os.path.join(root, name).replace(self.root_dir, '')
                self.labels.append(temp.split('/')[0])
                self.im_paths.append(temp)

    def __len__(self):
        return len(self.im_paths) * 4

    def __getitem__(self, idx):
        actual_idx = int(idx / 4)
        rotation = idx % 4
        im_name = os.path.join(self.root_dir, self.im_paths[actual_idx])
        im = cv2.imread(im_name)
        im = cv2.resize(im, (200, 200))
        label = label_to_num[self.labels[actual_idx]]
        
        # Get image info
        (h, w) = im.shape[:2]
        center = (w / 2, h / 2)
        rotate_map = {1: 90, 2: 180, 3: 270} # `rotation` -> degrees
        scale = 1.0
        
        if rotation != 0:
            # Perform counter clockwise rotation
            M = cv2.getRotationMatrix2D(center, rotate_map[rotation], scale)
            im = cv2.warpAffine(im, M, (h, w))

        im = torch.from_numpy(im.reshape((3, 200, 200))).type(torch.FloatTensor)
        return im, label


# Data label format: num-color-fill-shape
train_data = SetDataset('data/train/')
test_data = SetDataset('data/dev/')
# test_data = SetDataset('data/test/')


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
    

# def train(model, device, train_loader, optimizer, epoch, log_interval):
def train(model, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        # data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# def test(model, device, test_loader, return_images=False, log_interval=None):
def test(model, test_loader, return_images=False, log_interval=None):
    model.eval()
    test_loss = 0
    correct = 0

    correct_images = []
    correct_values = []

    error_images = []
    predicted_values = []
    gt_values = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            # data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss_on = model.loss(output, label, reduction='sum').item()
            test_loss += test_loss_on
            pred = output.max(1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            if return_images:
                if num_correct > 0:
                    correct_images.append(data[correct_mask, ...].data.cpu().numpy())
                    correct_value_data = label[correct_mask].data.cpu().numpy()[:, 0]
                    correct_values.append(correct_value_data)
                if num_correct < len(label):
                    error_data = data[~correct_mask, ...].data.cpu().numpy()
                    error_images.append(error_data)
                    predicted_value_data = pred[~correct_mask].data.cpu().numpy()
                    predicted_values.append(predicted_value_data)
                    gt_value_data = label[~correct_mask].data.cpu().numpy()[:, 0]
                    gt_values.append(gt_value_data)
            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))
    if return_images:
        correct_images = np.concatenate(correct_images, axis=0)
        error_images = np.concatenate(error_images, axis=0)
        predicted_values = np.concatenate(predicted_values, axis=0)
        correct_values = np.concatenate(correct_values, axis=0)
        gt_values = np.concatenate(gt_values, axis=0)

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    if return_images:
        return correct_images, correct_values, error_images, predicted_values, gt_values
    else:
        return test_accuracy


BATCH_SIZE = 64
TEST_BATCH_SIZE = 50
EPOCHS = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.9
USE_CUDA = False
PRINT_INTERVAL = 100
WEIGHT_DECAY = 0.0005
MODEL_PATH = 'models'
# LOG_PATH = 'imagenet_full/' + 'log.pkl'
use_cuda = USE_CUDA and torch.cuda.is_available()

# device = torch.device("cuda" if use_cuda else "cpu")
# print('Using device', device)
import multiprocessing
print('num cpus:', multiprocessing.cpu_count())

kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

model = TinyNet()#.to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# start_epoch = model.load_last_model(CHECKPOINT_PATH)
start_epoch = 1

try:
    best_accuracy = 0
    for epoch in range(start_epoch, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch, PRINT_INTERVAL)
        test_accuracy = test(model, test_loader, False)
        if test_accuracy > best_accuracy:
            model.save_model(MODEL_PATH + '/%03d.pt' % epoch)    
            best_accuracy = test_accuracy

except KeyboardInterrupt as ke:
    print('Interrupted')
except:
    import traceback
    traceback.print_exc()
finally:
    # model.save_model(MODEL_PATH + '/%03d.pt' % epoch)
    pass
