"""
Created on Thu Mar  2 18:26:25 2023

@author: ksenijakovalenka
"""
import numpy as np
import torch # torch tensors
import torch.nn as nn # predifined layers and activations
from data_load_fully_connected import load_data_flat # my simplified version of reading in the file
from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset, DataLoader # class base and batch split

# torch
import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# quantum
import pennylane as qml
from pennylane import numpy as np

# Hyper-parameters 
input_size = 484 
hidden_size = 300
num_classes = 2
num_epochs = 50
batch_size = 100
learning_rate = 0.001

# array for cost function
loss_array = np.zeros([])
steps_array = np.zeros([])

class HoppingDataset(Dataset):

    def __init__(self, train=True):
        # Initialize data
        # read with separate module
        hoppings, phases = load_data_flat()
        
        # trining vs testing data set
        self.train = train
        
        # split into train set and test set
        hoppings_train, hoppings_test, phases_train, phases_test = train_test_split(hoppings, phases, test_size=0.2, random_state=42)
        
        # convert the requited bit to torch tensors and set as an attribute
        if(self.train):
            self.x_data = torch.from_numpy(hoppings_train.astype(np.float32)) # size [n_samples, n_features]
            self.y_data = torch.from_numpy(phases_train) # size [n_samples, 1]
            # access the number of data points
            self.n_samples = hoppings_train.shape[0]
            print(hoppings_train.shape[0])
        else:
            self.x_data = torch.from_numpy(hoppings_test.astype(np.float32)) # size [n_samples, n_features]
            self.y_data = torch.from_numpy(phases_test) # size [n_samples, 1]
            # access the number of data points
            self.n_samples = hoppings_test.shape[0]
            print(hoppings_test.shape[0])

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

# test out the loading
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
print(example_data.size())
print(example_targets.size())

"""
design quantum circuit 
"""
n_qubits = 2
#dev = qml.device("default.qubit", wires=n_qubits, shots=1024)  # shots to make it behave like quantum comuter
dev = qml.device("default.qubit", wires=n_qubits, shots=1024)

#@qml.qnode(dev, diff_method="parameter-shift") # need parameter shift rule, otherwise classical
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))  # inputs are passed as a rotation angles of each qubit, default roatation=X
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits)) # just CNOTs connectong all available qubits
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)] # <state|z|state>

# make Torch think it's a normal layer
n_layers = 1 # number of quantum layers
weight_shapes = {"weights": (n_layers, n_qubits)} # defines trainable parameters for the quantum circuit part

qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# Fully connected neural network with one hidden layer
class Hybrid(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Hybrid, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, num_classes)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.qlayer(out)

        # no activation and no softmax at the end
        return out

model = Hybrid(input_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (hoppings, labels) in enumerate(train_loader):  
        
        # convert labels in the desirable form
        labels = labels.reshape((batch_size))
        labels = labels.long()
        # Forward pass
        outputs = model(hoppings)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 40 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        
        steps_array = np.append(steps_array, i + epoch*8000/batch_size)
        loss_array = np.append(loss_array, loss.item())

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for hoppings, labels in test_loader:
        
        # convert labels in the desirable form
        labels = labels.reshape((batch_size))
        labels = labels.long()
        outputs = model(hoppings)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    print(n_correct)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 2000 test hoppings: {acc} %')

with torch.no_grad():

    # write loss into a file
    np.savetxt("fc_q_q_long.txt", loss_array)
    np.savetxt("q_batchnumber_new.txt", steps_array)

