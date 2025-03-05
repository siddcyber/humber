# Instructions
# 1. An Image Folder“ HAR_Images” is provided. It has three classes.
# 2. Using Pytorch, design Artificial Neural Network (ANN) to classify the categories of HAR_Images dataset.
# 3. Calculate the training and testing accuracies.
# 4. Plot training loss vs epochs.
# 5. Plot training and testing accuracies against epochs.
# 6. Your accuracy should be close to 95%.


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader, dataset , random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# first we need to load the data and standardize some transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
data = datasets.ImageFolder(root="HAR_Images", transform=transform)

# now lets split the data into training and testing
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = random_split(data, [train_size, test_size])

# now we need to create the dataloaders for the training and testing data
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# lets try to use the code the stupid professor gave us

# ANNiris = nn.Sequential(
#           nn.Linear(4,32),
#           nn.ReLU(),
#           nn.Linear(32,32),
#           nn.ReLU(),
#           nn.Linear(32,3),
#           )

ANNiris = nn.Sequential(
                nn.Flatten(),  # Flatten the image (3 x 64 x 64) into a vector.
                nn.Linear(3 * 64 * 64, 512),  # First hidden layer.
                nn.ReLU(),  # Activation.
                nn.Linear(512, 128),  # Second hidden layer.
                nn.ReLU(),  # Activation.
                nn.Linear(128, len(data.classes))  # Output layer.
).to(device)
# learningRate = 0.01

lossfunc = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(ANNiris.parameters(), lr=learningRate)
# changing the optimizer to Adam because SGD did not work
optimizer = optim.Adam(ANNiris.parameters(), lr=0.001)

epochs = 20
losses = torch.zeros(epochs)  # setting place holder for for loop

# lists to record results
train_losses = []
train_accuracies = []
test_accuracies = []
#  no need to manuallu convert the data as flatten does it for us
# for i, j in train_loader:
#     temp_data = i.view(-1, 4)
#     labels = j

for epoch in range(epochs):
    ANNiris.train() # Setting the model to training mode
    # these help track the losses and accuracy for each epoch
    epoch_loss = 0.0
    correct_train = 0
    total_train = 0
    # adding a loop to get the data from the train_loader and using NN.Squential to get required input
    for images, labels in train_loader:
        # first we need to move the data to the device
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        ypred = ANNiris(images)
        loss = lossfunc(ypred, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        # get the predicted class by getting the index of the max value in the output
        _, predicted = torch.max(ypred, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    # lets calculate the loss and accuracy for the epoch
    train_loss = epoch_loss / total_train
    train_accuracy = correct_train / total_train
    # finally we append the results to the lists
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.2f}, Train Accuracy: {train_accuracy:.2f}")
