# -*- coding: utf-8 -*-
"""train_b.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rPjFhSADQfAe4W_x2AX27sWcWnmz3We8
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sys
from skimage import io, transform

import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

from IPython.display import Image

# DataLoader Class
# if BATCH_SIZE = N, dataloader returns images tensor of size [N, C, H, W] and labels [N]
class ImageDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform=None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [labels,data] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,1:].to_numpy()
            labels = data.iloc[:,0].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
            labels = None
        
        self.images = images
        self.labels = labels
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads image of the given index and performs preprocessing.
        
        INPUT: 
        idx: index of the image to be loaded.
        
        OUTPUT:
        sample: dictionary with keys images (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape((32, 32, 3),order="F")
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
        
        sample = {"images": image, "labels": label}
        return sample

# Data Loader Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
train_data=sys.argv[1]
test_data=sys.argv[2]
model_path=sys.argv[3]
acc_path=sys.argv[5]
loss_path=sys.argv[4]
# Train DataLoader
#train_data = "" # Path to train csv file
#train_data = "/content/drive/MyDrive/A2.2-Data/train_data.csv"
#test_data = "/content/drive/MyDrive/A2.2-Data/public_test.csv"
train_dataset = ImageDataset(data_csv = train_data, train=True, img_transform=img_transforms)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

# Test DataLoader
#test_data = "" # Path to test csv file
#train_data = "/content/drive/MyDrive/A2.2-Data/train_data.csv"
#test_data = "/content/drive/MyDrive/A2.2-Data/public_test.csv"
test_dataset = ImageDataset(data_csv = test_data, train=True, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1)
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm2d(512)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.conv3 = nn.Conv2d(64,512,3,1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(512,1024,2,1)
        self.fc1 = nn.Linear(1024*1*1, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout=nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x)))) # -> n, 32, 15, 15
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))  # -> n, 64, 6, 6
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))  # -> n, 512, 2, 2
        x = F.relu(self.conv4(x))  # -> n, 1024, 1, 1
        x = x.view(-1, 1024)            # -> n, 1024
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)             
        return x

torch.manual_seed(51)
model = NeuralNetwork().to(device)
num_epochs=18
learning_rate=1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(sum(p.numel() for p in model.parameters()))
n_total_steps = len(train_loader)
avg_loss=[]
accuracy=[]
best_acc=0
for epoch in range(num_epochs):
    total_loss=0
    i=0
    for batch_idx, sample in enumerate(train_loader):
        i=i+1
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = sample['images'].to(device)
        labels = sample['labels'].to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss+=loss.item()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss+=[total_loss/i]
    model.eval()
    with torch.no_grad():
      n_correct = 0
      n_samples = 0
      n_class_correct = [0 for i in range(10)]
      n_class_samples = [0 for i in range(10)]
      for batch, samp in enumerate(test_loader):
          images = samp['images'].to(device)
          labels = samp['labels'].to(device)
          outputs = model(images)
        # max returns (value ,index)
          _, predicted = torch.max(outputs, 1)
          n_samples += labels.size(0)
          n_correct += (predicted == labels).sum().item()
        
          for i in range(batch):
              label = labels[i]
              pred = predicted[i]
              if (label == pred):
                  n_class_correct[label] += 1
              n_class_samples[label] += 1

      acc = n_correct / n_samples
      accuracy+=[acc]
      if(acc>best_acc):
        best_acc=acc
        torch.save(model.state_dict(), model_path)
    model.train()

loss_path="/content/drive/MyDrive/A2.2-Data/loss.txt"
acc_path="/content/drive/MyDrive/A2.2-Data/acc.txt"
np.savetxt(loss_path,avg_loss)
np.savetxt(acc_path, accuracy)

torch.save(model.state_dict(), "/content/drive/MyDrive/A2.2-Data/model.pth")
#torch.save(model.state_dict(), model_path)