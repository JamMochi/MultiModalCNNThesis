import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from torchvision import transforms 
from model import ProjectNet, ParallelData, matts_algorithm_gpu

parser =  argparse.ArgumentParser(description='Multi-Modal Projection Net')
parser.add_argument('--batch_size', type=int, default=32, help='intialize size of each batch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for optimizers')
parser.add_argument('--epochs', type=int, default=100, help='intialize the number of epochs')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes in dataset')
parser.add_argument('--fully_connected', type=bool, default=False, help='intialize fusion layer to be fully-connected')
parser.add_argument('--gpu', type=bool, default=False, help='utilize gpu')
parser.add_argument('--shape', type=tuple, default=(1, 128, 7, 7), help='dimension of fusion layer')
parser.add_argument('--file', type=str, default='d2_t3.txt', help='file output name')
parser.add_argument('--param', type=bool, default=False, help='intialize model weights with pretrained models')
parser.add_argument('--freeze', type=bool, default=False, help='freeze pretrainted model parameters')
parser.add_argument('--weighted_sum', type=bool, default=False, help='sum of the fusion weights bounded between 0 - 1')
parser.add_argument('--file_red', type=str, default='params/red/d3_red_weights.pt', help='weight file for rgb')
parser.add_argument('--file_green', type=str, default='params/green/d3_green_weights.pt', help='weight file for inf')
parser.add_argument('--file_blue', type=str, default='params/blue/d3_blue_weights.pt', help='weight file for rgb')
parser.add_argument('--file_inf', type=str, default='params/d3_inf_1000.pt', help='weight file for inf')
parameters = parser.parse_args()
print(parameters)

# DEVICE CONFIGURATION
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DATA PIPELINE------------------------------------------------------------------------------------
# DATASET PATHS------------------------------------------------------------------------------------
training_rgb = os.getcwd() + '\\dataset_three\\1000_rgb\\'
testing_rgb = os.getcwd() + '\\dataset_three\\test_rgb\\'

training_inf = os.getcwd() + '\\dataset_three\\1000_inf\\'
testing_inf = os.getcwd() + '\\dataset_three\\test_inf\\'

# DATA TRANSFORMATION-------------------------------------------------------------------------------
training_transform = transforms.Compose([transforms.Resize([70,70]), transforms.ToTensor()])
testing_transform = transforms.Compose([transforms.Resize([70,70]), transforms.ToTensor()])

# CREATE DATASET AND DATALOADERS--------------------------------------------------------------------
training_data = ParallelData(training_rgb, training_inf, training_transform, 10000)
dataloader = torch.utils.data.DataLoader(training_data, parameters.batch_size, 10000)

testing_data = ParallelData(testing_rgb, testing_inf, testing_transform)
testloader = torch.utils.data.DataLoader(testing_data, parameters.batch_size)
#---------------------------------------------------------------------------------------------------
# INTIALIZE MODEL AND MODEL PARAMETERS--------------------------------------------------------------
model = ProjectNet(parameters.shape, parameters.num_classes, parameters.fully_connected,  parameters.freeze)

# MOVE MODEL TO GPU---------------------------------------------------------------------------------
if parameters.gpu:
    model = model.to('cuda')

# LOAD IN PRETRAINED MODELS PARAMETERS INTO OUR CURRENT MODEL---------------------------------------
if parameters.param:
    # load pretrained models from the given file path. pretrained_dict are state_dictionary of the
    # pretrainted models
    pretrained_dict_red = torch.load(parameters.file_red, map_location=lambda storage, loc:storage)
    pretrained_dict_green = torch.load(parameters.file_green, map_location=lambda storage, loc:storage)
    pretrained_dict_blue = torch.load(parameters.file_blue, map_location=lambda storage, loc:storage)
    pretrained_dict_inf = torch.load(parameters.file_inf, map_location=lambda storage, loc:storage)

    # laod the state dictionary of our own model. 
    pre_net_one_red = model.pre_net_one.state_dict()
    pre_net_two_green = model.pre_net_two.state_dict()
    pre_net_three_blue = model.pre_net_three.state_dict()
    pre_net_four_inf = model.pre_net_four.state_dict()
    post_net_dict = model.post_net.state_dict()
    
    # retrieve all the values from the pretrained dictionary that associate with our dictionary
    filter_dict_red = {k: v for k, v in pretrained_dict_red.items() if k in pre_net_one_red}
    filter_dict_green = {k: v for k, v in pretrained_dict_green.items() if k in pre_net_two_green}
    filter_dict_blue = {k: v for k, v in pretrained_dict_blue.items() if k in pre_net_three_blue}
    filter_dict_inf = {k: v for k, v in pretrained_dict_inf.items() if k in pre_net_four_inf}
    filter_dict_post = {k: v for k, v in pretrained_dict_inf.items() if k in post_net_dict}

    # update our model dictionary
    pre_net_one_red.update(filter_dict_red)
    pre_net_two_green.update(filter_dict_green)
    pre_net_three_blue.update(filter_dict_blue)
    pre_net_four_inf.update(filter_dict_inf)
    post_net_dict.update(filter_dict_post)

    # load the model dictionary back into our original model
    model.pre_net_one.load_state_dict(pre_net_one_red)
    model.pre_net_two.load_state_dict(pre_net_two_green)
    model.pre_net_three.load_state_dict(pre_net_three_blue)
    model.pre_net_four.load_state_dict(pre_net_four_inf)
    model.post_net.load_state_dict(post_net_dict)

# MODEL LOSS FUNCTIONS AND OPTIMIZER-----------------------------------------------------------------
loss_function =  nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=parameters.learning_rate)


# TRAINING THE MODEL---------------------------------------------------------------------------------
f = open(parameters.file,'w')
f.write('Begin Training')
for epoch in range(parameters.epochs):
    
    # intialize cummulative training losee for the given epoch
    training_loss = 0.0
    model.train()
    
    for i, data in enumerate(dataloader, 0):
        # retrieve information from the dataloader
        inp, inp2, target = data
        red = inp[:,0:1,:,:]
        green = inp[:,1:2,:,:]
        blue = inp[:,2:3,:,:]
        
        if parameters.gpu:
            red = red.to('cuda')
            green = green.to('cuda')
            blue = blue.to('cuda')
            inp2 = inp2.to('cuda')
            target = target.to('cuda')
            
        # input values into our model
        optimizer.zero_grad()
        outputs = model(red, green, blue, inp2)
        
        # calculate the loss with the given output with our target
        loss = loss_function(outputs, target)
        
        # calculate the gradient and update our parameters
        loss.backward()
        optimizer.step()
        
        # use matt's algorithm for test number 5
        if parameters.weighted_sum:
            matts_algorithm_gpu(model)

        training_loss += loss.item()
        
    average_loss = training_loss/len(dataloader.dataset)
    print(f'Average loss on Epoch {epoch}: {average_loss}')
    f.write(f'Average loss on Epoch {epoch}: {average_loss}\n')

    f.write('Begin Testing the Model')
    print('Begin Testing the Model')

    test_loss = 0.0

    class_correct = list(0. for i in range(parameters.num_classes))
    class_total = list(0. for i in range(parameters.num_classes))
 
    model.eval()
 
    for i, data in enumerate(testloader, 0):
        inp, inp2, target = data
        red = inp[:,0:1,:,:]
        green = inp[:,1:2,:,:]
        blue = inp[:,2:3,:,:]
    
        if parameters.gpu:
            red = red.to('cuda')
            green = green.to('cuda')
            blue = blue.to('cuda')
            inp2 = inp2.to('cuda')
            target = target.to('cuda')
    
        outputs = model(red, green, blue, inp2)
        loss = loss_function(outputs, target)
        test_loss += loss.item()

        _, pred = torch.max(outputs, 1)
        class_correct = (pred == target).sum().item()

    print(f'Average Test Error: {test_loss/target.shape[0]}')
    f.write(f'Average Test Error: {test_loss/target.shape[0]}')

    print(f'Total Accuracy: {100 * class_correct / target.shape[0]}')
    f.write(f'Total Accuracy: {100 * class_correct / target.shape[0]}')

print('Training is Complete')
f.write('Training is Complete')
torch.save(model.state_dict(),'/home/w004hsn/CNNTrainer/fully.pt')

#=============================================================================
