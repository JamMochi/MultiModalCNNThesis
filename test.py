import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms 
from model import ProjectNet, ParallelData, matts_algorithm

parser =  argparse.ArgumentParser(description='Multi-Modal Projection Net')
parser.add_argument('--batch_size', type=int, default=32, help='intialize size of each batch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for optimizers')
parser.add_argument('--epochs', type=int, default=500, help='intialize the number of epochs')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes in dataset')
parser.add_argument('--dropout', type=bool, default=False, help='allow for dropout')
parser.add_argument('--fully_connected', type=bool, default=False, help='intialize fusion layer to be fully-connected')
parser.add_argument('--gpu', type=bool, default=True, help='utilize gpu')
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

testing_inf = '/home/w004hsn/CNNTrainer/dataset_three/test_inf/'
testing_rgb = '/home/w004hsn/CNNTrainer/dataset_three/test_rgb/'

testing_transform = transforms.Compose([transforms.Resize([70,70]), transforms.ToTensor()])

testing_data = ParallelData(testing_rgb, testing_inf, testing_transform)
testloader = torch.utils.data.DataLoader(testing_data, 100)

# SHAPE, NUM_CLASSES, FC, DROPOUT, FREEZE
model = ProjectNet((1, 128, 7, 7), 3, True, True, False)
import_model = torch.load('fully.pt', map_location=lambda storage, loc:storage)
model.load_state_dict(import_model)

fusion_layer = model.fuse.state_dict()

zero_state = {'feature_weight_one': torch.zeros([1, 6272])}

#fusion_dict = model.fuse.state_dict()
filter_dict_fusion = {k: v for k, v in zero_state.items() if k in fusion_layer}
fusion_layer.update(filter_dict_fusion)
model.fuse.load_state_dict(fusion_layer)

model = model.to('cuda')

loss_function =  nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=parameters.learning_rate)

test_loss = 0.0

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
print(f'Total Accuracy: {100 * class_correct / target.shape[0]}')
print('Training is Complete')
