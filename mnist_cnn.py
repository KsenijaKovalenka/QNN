"""
Hybrid nn for written digit recognition
"""

# imports
import matplotlib.pyplot as plt
import numpy as np

# torch
import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

"""
MNIST data load
"""
# train data

# Concentrating on the first n samples
n_samples = 500
batches = 10
learning_rate = 0.001
input_size = 784 
num_classes = 2
num_epochs = 5
hidden_size = 300

# stre the weights and initialisations
epoch_array_size = int((n_samples*2/batches))
initialisations_array_1 = np.zeros(epoch_array_size*num_epochs)
initialisations_array_2 = np.zeros(epoch_array_size*num_epochs)
weights_array_1 = np.zeros(epoch_array_size*num_epochs)
weights_array_2 = np.zeros(epoch_array_size*num_epochs)
steps_array = np.zeros(epoch_array_size*num_epochs)
label_array = np.zeros((epoch_array_size*num_epochs, batches))

# array for cost function
loss_array = np.zeros([])

X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

# Leaving only labels 0 and 1 
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=batches, shuffle=True)

# n_samples_show = 6

# data_iter = iter(train_loader)
# fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

# while n_samples_show > 0:
#     images, targets = data_iter.__next__()

#     axes[n_samples_show - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
#     axes[n_samples_show - 1].set_xticks([])
#     axes[n_samples_show - 1].set_yticks([])
#     axes[n_samples_show - 1].set_title("Labeled: {}".format(targets.item()))
    
#     n_samples_show -= 1
# plt.savefig('inputs.png', dpi=300)

# test data
n_samples_t = 50

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples_t], 
                np.where(X_test.targets == 1)[0][:n_samples_t])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=batches, shuffle=True)

# reshaping test 
examples = iter(test_loader)
example_data, example_targets = next(examples)
print(example_data.size())  # will proably need to flatten it
example_data = torch.reshape(example_data, (batches,-1))
print(example_data.size())
print(example_targets)


# Fully connected neural network with one hidden layer
class Hybrid(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Hybrid, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        # no activation and no softmax at the end
        return out

model = Hybrid(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (hoppings, labels) in enumerate(train_loader):  
        
        # convert labels in the desirable form
        hoppings = torch.reshape(hoppings, (batches,-1))
        labels = labels.reshape((batches))
        labels = labels.long()
        label_array[i + epoch_array_size*epoch,:] = labels
        # Forward pass
        outputs = model(hoppings)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps_array[i + epoch_array_size*epoch] = i + epoch_array_size*epoch
        
        
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        
        loss_array = np.append(loss_array, loss.item())

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for hoppings, labels in test_loader:
        
        # convert labels in the desirable form
        hoppings = torch.reshape(hoppings, (batches,-1))
        labels = labels.reshape((batches))
        labels = labels.long()
        outputs = model(hoppings)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    print(n_correct)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 100 test hoppings: {acc} %')


# plot the change in circuit parametters
with torch.no_grad():

    # write loss into a file
    np.savetxt("fc_c_c.txt", loss_array)
    # np.savetxt("c_batchnumber.txt", steps_array)
