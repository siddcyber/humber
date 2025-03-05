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
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# lets try to use the code the stupid professor gave us

# ANNiris = nn.Sequential(
#           nn.Linear(4,32),
#           nn.ReLU(),
#           nn.Linear(32,32),
#           nn.ReLU(),
#           nn.Linear(32,3),
#           )

ANNiris = nn.Sequential( nn.Flatten(),  # Flatten the image (3 x 64 x 64) into a vector.
                nn.Linear(3 * 64 * 64, 512),  # First hidden layer.
                nn.ReLU(),  # Activation.
                nn.Linear(512, 128),  # Second hidden layer.
                nn.ReLU(),  # Activation.
                nn.Linear(128, len(data.classes))  # Output layer.
)
learningRate = 0.01
lossfunc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ANNiris.parameters(), lr=learningRate)

epochs = 2001
losses = torch.zeros(epochs)  # setting place holder for for loop.
for i, j in train_loader:
    temp_data = i.view(-1, 4)
    labels = j

for epoch in range(epochs):

    ypred = ANNiris(temp_data.float())
    loss = lossfunc(ypred, labels)
    losses[epoch] = loss.detach()

    if (epoch % 100) == 0:
        print(f' epochs : {epoch}  loss : {loss : 2.2f}')

    # backprop
    optimizer.zero_grad()  # Initializing the gradient to zero. zero_grad() restarts looping without losses from the last
    # step if you use the gradient method for decreasing the error (or losses).If you do not use
    # zero_grad() the loss will increase not decrease as required.Gradients accumulate with every backprop.
    # To prevent compounding we need to  reset the stored gradient for each new epoch.

    loss.backward()  # do gradient of all parameters for which we set required_grad= True. parameters could be any
    # variable defined in code.

    optimizer.step()  # according to the optimizer function (defined previously in our code), we update those parameters
    # to finally get the minimum loss(error).

accuracy = 100 * torch.mean((torch.argmax(ypred, axis=1) == labels).float())


