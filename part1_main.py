import os
import time
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
import pickle
device = "cpu"
torch.set_num_threads(4)

### track loss, training time per iteration from 1 >= iteration < 40 ###
training_data = {} # format of {iteration (int): [batch_idx (int), loss (float), time (float)]}

batch_size = 256 # batch for one node
def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    ### I added this ###
    model.train()  # Set the model to training mode
    running_loss = 0.0
    total_time = 0.0
    current_data_iteration = 0
    ### I added this ###

    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your code goes here!
#########################################
########### group code begins ###########
#########################################
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Zero the parameter gradients

        start_time = time.time()  # Start time for each iteration

        # Forward pass
        output = model(data) # (nn.Module) -> Callable -> implicitly calls .forward(x) -> output
        loss = criterion(output, target)
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        end_time = time.time()  # End time for each iteration
        iteration_time = end_time - start_time
        total_time += iteration_time

        running_loss += loss.item()
        if batch_idx % 20 == 19:  # Print every 20 mini-batches
            current_data_iteration += 1
            print('[Epoch: %d, Batch: %5d] Loss: %.3f' %(epoch, batch_idx + 1, running_loss / 20))
            training_data[current_data_iteration] = [batch_idx, running_loss/20, iteration_time]

    average_time_per_iteration = total_time / len(train_loader)
    print("Average time per iteration for all 40 iterations:", average_time_per_iteration)

#########################################
############ group code ends ############
#########################################
    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main():
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    shuffle=True,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)
    
    print('-'*120)
    print(f'iteration / batch_idx / loss / iteration_time')
    
    with open('part-1.pkl', 'wb') as handle:
        pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for k,v in training_data.items():
        print(f'{k:<3} {v[0]:<4} {v[1]:.2f} {v[2]:.2f}')
    print('-'*120)


    print('finished')


if __name__ == "__main__":
    main()
