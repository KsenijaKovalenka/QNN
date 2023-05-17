#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:35:25 2023
Convolutional neural network
@author: ksenijakovalenka
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader # class base and batch split
from data_load import load_data # my simplified version of reading in the file
from sklearn.model_selection import train_test_split 

# Hyper-parameters 
num_epochs = 20
batch_size = 50
learning_rate = 0.01

# array for cost function
loss_array = np.zeros([])

class HoppingDataset(Dataset):

    def __init__(self, train=True):
        # Initialize data
        # read with separate module
        hoppings, phases = load_data()
        
        # trining vs testing data set
        self.train = train
        
        # split into train set and test set
        hoppings_train, hoppings_test, phases_train, phases_test = train_test_split(hoppings, phases, test_size=0.4, random_state=42)
        
        # convert the requited bit to torch tensors and set as an attribute
        if(self.train):
            self.x_data = torch.from_numpy(hoppings_train.astype(np.float32)) # size [n_samples, n_features]
            self.y_data = torch.from_numpy(phases_train) # size [n_samples, 1]
            # access the number of data points
            self.n_samples = hoppings_train.shape[0]
        else:
            self.x_data = torch.from_numpy(hoppings_test.astype(np.float32)) # size [n_samples, n_features]
            self.y_data = torch.from_numpy(phases_test) # size [n_samples, 1]
            # access the number of data points
            self.n_samples = hoppings_test.shape[0]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset
train_dataset = HoppingDataset()
test_dataset = HoppingDataset(train=False)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
examples = iter(test_loader)
example_data, example_targets = next(examples)

classes = ('trivial', 'topological')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 3)
        self.pool = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(6 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # -> n, 4, 11, 11 (n - batch size)
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 4, 4
        x = x.view(-1, 6 * 4 * 4)            # -> n, 96
        x = F.relu(self.fc1(x))               # -> n, 100
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 2
        return x

# check the sizes
model = ConvNet()
conv = model.conv1
pool = model.pool
x = example_data
print(x.size())
x = conv(x)
print(x.size())
x = pool(x)
print(x.size())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        # Forward pass
        outputs = model(images)
        
        # convert labels in the desirable form
        labels = labels.reshape((batch_size))
        labels = labels.long()
        
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        loss_array = np.append(loss_array, loss.item())

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        
        # convert labels in the desirable form
        labels = labels.reshape((batch_size))
        labels = labels.long()
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]

            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    print(n_correct)
    for i in range(2):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
        print(n_class_correct[i])
        

with torch.no_grad():

    # write loss into a file
    np.savetxt("cnn_c1_q.txt", loss_array)
    # np.savetxt("c_batchnumber.txt", steps_array)
