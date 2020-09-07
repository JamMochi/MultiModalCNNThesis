import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms 
from model import DefaultCNN, mDataset
import os

parser =  argparse.ArgumentParser(description='Convolution Net')
parser.add_argument('--num_classes', type=int, default=3, help='number classes for output')
parser.add_argument('--batch_size', type=int, default=32, help='size of each batch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate of optimizers')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--dropout', type=bool, default=False, help='include dropout')
parser.add_argument('--gpu', type=bool, default=False, help='utilize gpu')
parser.add_argument('--inf', type=bool, default=False, help='grayscale')
parser.add_argument('--weight_file', type=str, default='w2_d2.txt', help='save weight file provide name')
parser.add_argument('--inp_size', type=int, default=1, help='input size')
parser.add_argument('--file', type=str, default='d2_t1.txt', help='size of fusion')
parser.add_argument('--color', type=str, default='R', help='provide either R, G, or B')

parameters = parser.parse_args()
print(parameters)

# DEVICE CONFIGURATION
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DATASET PATHS
if parameters.inf == True:
    root = f'{os.getcwd()}\\dataset_three\\1000_inf\\'
    test_root = f'{os.getcwd()}\\dataset_three\\test_inf\\'
else:
    root = f'{os.getcwd()}\\dataset_three\\1000_rgb\\'
    test_root = f'{os.getcwd()}\\dataset_three\\test_rgb\\'

# TRANSFORM PIPELINES
#------------------------------------------------------------------------------
# Transforms that are going to be utilize for the training dataset
training_transform = transforms.Compose([transforms.Resize([70,70]),
                                         transforms.ToTensor()])

testing_transform = transforms.Compose([transforms.Resize([70,70]), 
                                        transforms.ToTensor()])

training_data = mDataset(root, training_transform, inf=parameters.inf)                            # Run the training data through the data pipeline -> training_transform (30)
testing_data = mDataset(test_root, testing_transform, 10000)                                      # Run the testing data through the data pipeline -> testing_transforma (33)

dataloader = torch.utils.data.DataLoader(training_data, batch_size=parameters.batch_size)         # Create a dataloader that will create a batch for training and testing
testloader = torch.utils.data.DataLoader(testing_data, batch_size=parameters.batch_size)

# -----------------------------------------------------------------------------
# TRAINING THE MODEL
#------------------------------------------------------------------------------

network = DefaultCNN(dropout=parameters.dropout, 
                     num_classes=parameters.num_classes, 
                     inp_size=parameters.inp_size)

if parameters.gpu:
    network = network.to('cuda')                                                              # Create a Convolution Neural Network object

loss_function = nn.CrossEntropyLoss()                                                         # Declare the loss function that is going to be used for testing
optimizer = optim.SGD(network.parameters(), lr=parameters.learning_rate, momentum=0.9)        # Method of optimization (stochasic gradient descent)

f = open(parameters.file,'w')
f.write('Begin Training')                                                                     # Intialize the number of epochs that are going to be use for training
for epoch in range(parameters.epochs):                                                        # Run for NUM_OF_EPOCHS
    
    training_loss = 0.0                                                                       # Reset the training loss after each epoch
    network.train()                                                                           # Train the network
    
    for i, data in enumerate(dataloader, 0):                                                  # Retrieve data in batches from the dataloader
        inputs, labels = data
        
        if(parameters.color == "B"):
            inputs = inputs[:,2:3,:,:]
        elif(parameters.color == "G"):
            inputs = inputs[:,1:2,:,:]
        else:
            inputs = inputs[:,0:1,:,:]
            
        if parameters.gpu:                                                                    # Data returns two values (input_data, labels) store them into those variables
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

        optimizer.zero_grad()                                                                 # Initialize weights to be zero for the convolution neural network
        outputs = network(inputs)                                                             # Apply inputs into network. Forward propagation will happen during this step
    
        loss = loss_function(outputs, labels)                                                 # Calculate the loss after the forward propagation. (Loss function is declared on line 123)
        loss.backward()                                                                       # Perform Backpropagation
        optimizer.step()                                                                      # parameter updating aka weight updating
        training_loss += loss.item()                                                          # Cumulative loss for a given epoch
        
    average_loss = training_loss/len(dataloader.dataset)                                      # average loss = total loss/ number of datapoints
    print(f'Average loss on Epoch {epoch}: {average_loss}\n')
    f.write(f'Average loss on Epoch {epoch}: {average_loss}\n')
    f.write('Begin Testing the Model')
    print('Begin model testing')
    test_loss = 0.0                                                                           # Intialize Testing Loss

    class_correct = list(0. for i in range(parameters.num_classes))                           # Create a binary list to story numb$
    class_total = list(0. for i in range(parameters.num_classes))                             # Create a binary list to story numb$

    network.eval()

    for i, data in enumerate(testloader, 0):
        inp, target = data
        inp = inp[:,0:1,:,:]
        
        if(parameters.color == "B"):
            inputs = inputs[:,2:3,:,:]
        elif(parameters.color == "G"):
            inputs = inputs[:,1:2,:,:]
        else:
            inputs = inputs[:,0:1,:,:]
            
        if parameters.gpu:
            inp = inp.to('cuda')
            target = target.to('cuda')
    
        outputs = network(inp)
        loss = loss_function(outputs, target)
        test_loss += loss.item()

        _, pred = torch.max(outputs, 1)
        class_correct = (pred == target).sum().item()
    
    print(f'Average Test Error: {test_loss/target.shape[0]}')
    f.write(f'Average Test Error: {test_loss/target.shape[0]}')
    
    print(f'Total Accuracy: {100 * class_correct / target.shape[0]}')
    f.write(f'Total Accuracy: {100 * class_correct / target.shape[0]}')     

print("Training Complete")
f.write('Training is Complete')
torch.save(network.state_dict(), f'{os.getcwd()}/{parameters.weight_file}')

    
