"""
Hybrid nn for written digit recognition
"""

# imports
import matplotlib.pyplot as plt

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
        hoppings = torch.reshape(hoppings, (batches,-1))
        labels = labels.reshape((batches))
        labels = labels.long()
        label_array[i + epoch_array_size*epoch,:] = labels
        # Forward pass
        outputs = model(hoppings)

        with torch.no_grad():
            # get the weights
            required_weights = list(model.parameters())[2].data.numpy()
            weights_array_1[i + epoch_array_size*epoch] = required_weights[0][0]
            weights_array_2[i + epoch_array_size*epoch] = required_weights[0][1]
            # get the inputs
            interm_output = model.l1(hoppings)
            initialisations_array_1[i + epoch_array_size*epoch] = interm_output.data.numpy()[0][0]
            initialisations_array_2[i + epoch_array_size*epoch] = interm_output.data.numpy()[0][1]

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

# # Visualising what the circuit is doing after training
# with torch.no_grad():
#     # to check I'm outputing the correct thing
#     # for name, param in model.named_parameters():
#     #     print(name)
#     #     print(param)

#     required_weights = list(model.parameters())[2].data.numpy()
#     print(required_weights)
#     print(qml.draw(qnode, expansion_strategy="device")([0,0],required_weights))
#     fig1, ax1 = qml.draw_mpl(qnode)([0,0],required_weights)
#     fig1.show()


# print("--------------------")
# # run through 1 test data sample
# with torch.no_grad():
#     # reshaping test 
#     examples = iter(test_loader)
#     example_data, example_targets = next(examples)

#     print("target:")
#     print(example_targets)
#     example_data = torch.reshape(example_data, (batches,-1))
#     interm_output = model.l1(example_data)
#     print("in ql")
#     print(interm_output.data.numpy())
#     print("out ql")
#     output = model.qlayer(interm_output)
#     print(output)
#     smth, predicted = torch.max(output.data, 1)
#     print("network prediction")
#     print(smth)
#     print(predicted)
#     value1 = interm_output.data.numpy()[0][0]
#     value2 = interm_output.data.numpy()[0][1]
#     print(qml.draw(qnode, expansion_strategy="device")([value1, value2],required_weights))


# plot the change in circuit parametters
with torch.no_grad():

    # calculate mean and std for the batches
    label_mean = np.mean(label_array.numpy(), axis=1)
    label_std = np.std(label_array.numpy(), axis=1)
    label_mean_mean = np.mean(label_mean)
    label_mean_std = np.std(label_mean)

    # # plot averages and stds
    # fig, ax = plt.subplots()
    # ax.set_title("Sample Mean")
    # ax.scatter(steps_array, label_mean, s=0.7)
    # print(label_mean_mean)

    # fig, ax = plt.subplots()
    # ax.set_title("Sample Std")
    # ax.scatter(steps_array, label_std, s=0.7)
    # print(label_mean_std)



    # Map values to colors
    colors = {0. : 'red', 1. : 'blue'}  # map values to colors
    mapped_colors = [colors[label] for label in label_array.numpy()[:,0]]

    # plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle('Imbedding angles', fontsize=14)
    ax1.set_title("qubit 1", fontsize=14)
    ax2.set_title("qubit 2", fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_xlabel('batch number', fontsize=12)
    ax1.set_ylabel('angle (rad)', fontsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_xlabel('batch number', fontsize=12)
    ax1.scatter(steps_array, initialisations_array_1, c=mapped_colors, s=0.7)
    ax2.scatter(steps_array, initialisations_array_2, c=mapped_colors, s=0.7)
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('imbedding.png', dpi=300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Weight angles", fontsize=14)
    ax1.set_title("qubit 1", fontsize=14)
    ax2.set_title("qubit 2", fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_xlabel('batch number', fontsize=12)
    ax1.set_ylabel('angle (rad)', fontsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_xlabel('batch number', fontsize=12)
    ax1.scatter(steps_array, weights_array_1, c=mapped_colors, s=0.7)
    ax2.scatter(steps_array, weights_array_2, c=mapped_colors, s=0.7)
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('weights.png', dpi=300)

    # write loss into a file
    np.savetxt("fc_q_c.txt", loss_array)
    np.savetxt("c_batchnumber.txt", steps_array)

    X = [initialisations_array_1[-1],initialisations_array_2[-1]]
    Y = [[weights_array_1[-1],weights_array_2[-1]]]

    print(qml.draw(qnode, expansion_strategy="device")(X,Y))

    print("final label")
    print(label_array.numpy()[-1,0])
    print("final output")
    print(qnode(X,Y))
    print("Batch size")
    print(batches)


    # producing a finale figure
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(131)  # First set of axes
    ax2 = fig.add_subplot(132)  # Second set of axes
    ax3 = fig.add_subplot(133)  # Third set of axes
    ax1.set_title("Imbedding on qubit 1", fontsize=16)
    ax2.set_title("Imbedding on qubit 2", fontsize=16)
    ax3.set_title("Weights of rotation on qubit 1", fontsize=14)
    ax1.set_ylabel('angle (rad)', fontsize=16)
    ax1.set_xlabel('batch number', fontsize=16)
    ax2.set_xlabel('batch number', fontsize=16)
    ax3.set_xlabel('batch number', fontsize=16)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    ax1.scatter(steps_array, initialisations_array_1, c=mapped_colors, s=0.7)
    ax2.scatter(steps_array, initialisations_array_2, c=mapped_colors, s=0.7)
    ax3.scatter(steps_array, weights_array_1, c="cornflowerblue", s=0.7)
    plt.savefig('final_shallow.png', dpi=300)

    hist_plot = False
    if (hist_plot):
        # histograms
        means = np.zeros((num_epochs,2))
        stds = np.zeros((num_epochs,2))
        bin_value_array = np.zeros((epoch_array_size,2))
        for epoch in range(num_epochs):
            for batch_number in range(epoch_array_size):
                batch = label_array.numpy()[batch_number + epoch_array_size*epoch,:]
                # print(batch_number + epoch_array_size*epoch)
                bin_values, bin_boundaries = np.histogram(batch, bins=2, range=(0, 1))
                bin_value_array[batch_number] = bin_values
            print(bin_value_array)
            mean = np.mean(bin_value_array, axis=0)
            std = np.std(bin_value_array, axis=0)
            means[epoch] = mean
            stds[epoch] = std

        # plot the means
        fig, ax = plt.subplots()
        ax.set_title("Batch frequencies")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Batch mean frequency")
        ax.set_ylim(0, batches)
        ax.scatter(np.arange(1, num_epochs+1, 1), means[:,0], label="zeros")
        ax.scatter(np.arange(1, num_epochs+1, 1), means[:,1], label="ones")
        ax.errorbar(np.arange(1, num_epochs+1, 1), means[:,0], stds[:,0], linestyle='')
        ax.errorbar(np.arange(1, num_epochs+1, 1), means[:,1], stds[:,1], linestyle='')
        ax.legend()

        # histograms within epochs
        means = np.zeros((num_epochs,2))
        stds = np.zeros((num_epochs,2))
        bin_value_array = np.zeros((epoch_array_size,2))
        for epoch in range(num_epochs):
            for batch_number in range(epoch_array_size):
                batch = label_array.numpy()[batch_number + epoch_array_size*epoch,:]
                # print(batch_number + epoch_array_size*epoch)
                bin_values, bin_boundaries = np.histogram(batch, bins=2, range=(0, 1))
                bin_value_array[batch_number] = bin_values
            # print(bin_value_array)
            # mean = np.mean(bin_value_array, axis=0)
            # std = np.std(bin_value_array, axis=0)
            # means[epoch] = mean
            # stds[epoch] = std

            # plot the actual bin sizes for each epoch
            fig, ax = plt.subplots()
            ax.set_title("Batch frequencies")
            ax.set_xlabel(f"Epoch {epoch}")
            ax.set_ylabel("Batch frequency")
            ax.set_ylim(0, batches)
            ax.plot(np.arange(1, epoch_array_size+1, 1), bin_value_array[:,0], label="zeros", linewidth=0.5)
            ax.plot(np.arange(1, epoch_array_size+1, 1), bin_value_array[:,1], label="ones", linewidth=0.5)
            ax.legend()
