# Imports here
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

import json
import time
import os

print(len(os.listdir('flowers/valid')))
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


training_transforms = transforms.Compose([
    transforms.CenterCrop((224,224)),
    transforms.RandomRotation((45, 90)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transforms = test_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=training_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

print(train_data)
print(test_data)
print(valid_data)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model = models.densenet121(pretrained=True)
print(model)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
#     ('dropout', nn.Dropout(p=0.03)),
    # 50176 comes from 224 X 224, shape of the images
    ('fc1', nn.Linear(1024, 500)),
    ('relu1', nn.ReLU()),
#     ('fc2', nn.Linear(25139, 12620)),
#     ('relu2', nn.ReLU()),
#     ('fc1', nn.Linear(12620, 500)),
#     ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(500, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

criterion = nn.NLLLoss()
print(classifier.parameters)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.01)


def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    count = 0
    for images, labels in testloader:
        labels.resize_(images.shape[0])

        images, labels = inputs.to(device), labels.to(device)

        output = model.forward(images)
        print(output.size())
        print(labels.size())
        ##### PROBLEM batch size (818) is being reduced by size of batch (64, every time leaving 51 at the end
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.cuda.FloatTensor).mean()

    return test_loss, accuracy


# device = 'cuda'
# epochs = 2
# steps = 0
# running_loss = 0
# print_every = 40
# model.to(device)
# for e in range(epochs):
#     model.train()
#     for ii, (inputs, labels) in enumerate(trainloader):

# #         print('labels - ', labels)
# #         inputs.resize_(inputs.size()[0], 50176)
#         print(inputs.size())
#         # Move input and label tensors to the GPU
#         inputs, labels = inputs.to(device), labels.to(device)

#         start = time.time()

#         outputs = model.forward(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         if steps % print_every == 0:
#             # Make sure network is in eval mode for inference
#             model.eval()

#             # Turn off gradients for validation, saves memory and computations
#             with torch.no_grad():
#                 test_loss, accuracy = validation(model, validloader, criterion)

#             print("Epoch: {}/{}.. ".format(e+1, epochs),
#                   "Training Loss: {:.3f}.. ".format(running_loss/print_every),
#                   "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
#                   "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

#             running_loss = 0

#             # Make sure training is back on
#             model.train()


epochs = 3
print_every = 40
steps = 0

# change to cuda
model.to('cuda')

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1

        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e + 1, epochs),
                  "Loss: {:.4f}".format(running_loss / print_every))

            running_loss = 0