import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class mDataset(Dataset):
    
    """ 
    This class will be use as a datapipe line in retrieving data from 
    given root directory. 

    - Returns a dataset with given transformation that can be feed into
    a dataloader object. 
    
    """

    # Default constructor for the car dataset pipeline taking parameters 
    # that will specify if transfoing_rate 0.01rm is needed and check to see if its 
    # a infared (grayscale) image                          
    def __init__(self, root_dir, transform=None, inf=False):
        self.inf = inf                               
        self.data = os.listdir(root_dir)                                             
        self.root_dir = root_dir                                               
        self.tranformed = transform                                             
    
    # Returns the number of files for the given dataset                         
    def __len__(self):                                                          
        return len(self.data)                                               
    
    # Returns the tensor of the specify image                                  
    def __getitem__ (self, idx):                                                    
        imagePath = self.root_dir + self.data[idx]                          
        label = self.data[idx][1]
        sample = Image.open(imagePath)
        if self.inf == False:
            sample = sample.convert('RGB')                                         
        if self.tranformed:                                                     
            sample = self.tranformed(sample)
        return sample, int(label)-1     

class ParallelData(Dataset):
    
    """ 
    This class will be use as a datapipe line in retrieving data from 
    given root directory. This class will also take in two dataset of 
    different modals. The purpose of this dataset object is to return 
    the same image with two different modality.

    - Returns a dataset with given transformation that can be feed into
    a dataloader object. 
    
    """
    
    def __init__(self, data_one, data_two, transform=None):
        self.root_one = data_one
        self.root_two = data_two
        self.dataset_one = os.listdir(data_one)
        self.dataset_two = os.listdir(data_two)
        self.is_transform = transform
        
    def __len__(self):
        return len(self.dataset_one)
    
    def __getitem__(self, idx):
        label = self.dataset_one[idx][1]
        imagePath_one = self.root_one + self.dataset_one[idx]
        imagePath_two = self.root_two + self.dataset_two[idx]
        inp_one = Image.open(imagePath_one).convert('RGB')
        inp_two = Image.open(imagePath_two)
        if self.is_transform:
            inp_one = self.is_transform(inp_one)
            inp_two = self.is_transform(inp_two)
        return (inp_one, inp_two, int(label) - 1)
        
class FCube(nn.Module):
    ''' 
    Fusion Cube, creates a weight tensor with the same dimensions of the output feature maps
    from the pre_net. 
    '''
    def __init__(self, shape, fc):
        super(FCube, self).__init__()
        _, x, y, z = shape
        self.fc = fc
        if self.fc:
            self.f_layer = nn.Linear(in_features=4*128*7*7, out_features=128*7*7)
        else:
            self.feature_weight_one = nn.Parameter(torch.randn(x, y, z).uniform_(0,1).view(-1, 128*7*7), requires_grad=True)
            self.feature_weight_two = nn.Parameter(torch.randn(x, y, z).uniform_(0,1).view(-1, 128*7*7), requires_grad=True)
            self.feature_weight_three = nn.Parameter(torch.randn(x, y, z).uniform_(0,1).view(-1, 128*7*7), requires_grad=True)
            self.feature_weight_four = nn.Parameter(torch.randn(x, y, z).uniform_(0,1).view(-1, 128*7*7), requires_grad=True)
                
    def forward(self, m1_x, m2_x, m3_x, m4_x):
        m1_x = m1_x.view(-1, 128*7*7)
        m2_x = m2_x.view(-1, 128*7*7)
        m3_x = m3_x.view(-1, 128*7*7)
        m4_x = m4_x.view(-1, 128*7*7)
        
        if self.fc:
            m3_x = torch.cat((m1_x, m2_x, m3_x, m4_x),1)
            return self.f_layer(m3_x)
        else:
            return (self.feature_weight_one * m1_x) + (self.feature_weight_two * m2_x) + (self.feature_weight_three * m3_x) + (self.feature_weight_four * m4_x) 

class DefaultCNN(nn.Module):
    
    # The init function intialize the necessary layers that will be within     --------------------------------------------------------------> 
    # the network. 
    def __init__(self, dropout, num_classes, inp_size):
        super(DefaultCNN, self).__init__()
        self.dropout = dropout                                          
        self.conv1 = nn.Conv2d(in_channels=inp_size, out_channels=32, kernel_size=3)    
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)   
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        
        self.pool = nn.MaxPool2d(2, 2)                                          
        
        if self.dropout:
            p = 0.4
            self.drop_layer1 = nn.Dropout(p=p)
            self.drop_layer2 = nn.Dropout(p=p-0.15)
            self.drop_layer3 = nn.Dropout(p=p-0.10)
        
        self.fc1 = nn.Linear(in_features=128*7*7, out_features=128)             
        self.out = nn.Linear(in_features=128, out_features=num_classes)                    
        
    # 
    def forward(self, input_):                                                 
        
        input_ = self.pool(F.relu(self.conv1(input_)))                         
        if self.dropout:
            input_ = self.drop_layer1(input_)
        input_ = self.pool(F.relu(self.conv2(input_)))
        if self.dropout:
            input_ = self.drop_layer2(input_)                                      
        input_ = self.pool(F.relu(self.conv3(input_)))
        if self.dropout:
            input_ = self.drop_layer3(input_)
        
        input_ = input_.view(-1, 128*7*7)                                       
        input_ = F.relu(self.fc1(input_))
        input_ = self.out(input_)                                               
        return input_                                                           

            
class PreConv(nn.Module):
    '''
    A custom network that models the front portion of the convolution neural network. This part
    of the network will be a fully convolutional network.
    '''
    def __init__(self, start_filter):
        super(PreConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=start_filter, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x        

class PostConv(nn.Module):
    '''
    A custom network that models the end portion of the convolution neural network. This part of the
    network will contain all the linear layers.
    '''
    def __init__(self, num_classes):
        super(PostConv, self).__init__()
        self.fc1 = nn.Linear(in_features=128*7*7, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

class ProjectNet(nn.Module):
    '''
    A custom fusion network 
    '''
    def __init__(self, shape, num_classes, fc, freeze):
        super(ProjectNet, self).__init__()
        self.fc = fc
        self.pre_net_one = PreConv(start_filter=1)
        self.pre_net_two = PreConv(start_filter=1)
        self.pre_net_three = PreConv(start_filter=1)
        self.pre_net_four = PreConv(start_filter=1)
        
        self.fuse = FCube(shape, fc=self.fc)
        self.post_net = PostConv(num_classes)

        if freeze:
            parameter_freezing(self.pre_net_one)
            parameter_freezing(self.pre_net_two)
            parameter_freezing(self.pre_net_three)
            parameter_freezing(self.pre_net_four)

            
    def forward(self, red, green, blue, inf):
        
        m1_x = self.pre_net_one(red)
        m2_x = self.pre_net_two(green)
        m3_x = self.pre_net_three(blue)
        m4_x = self.pre_net_four(inf)
        
        fusion_map = self.fuse(m1_x, m2_x, m3_x, m4_x)
        
        output = self.post_net(fusion_map)
        
        return output

def parameter_freezing(model):
    for params in model.parameters():
        params.requires_grad = False 
        
#def matts_algorithm(model):
#    state_dict = model.state_dict()
#    t1_param = state_dict['fuse.feature_weight_one']
#    t2_param = state_dict['fuse.feature_weight_two']
#    f_weight_one, f_weight_two = tensor_maker(t1_param, t2_param)
#    state_dict['fuse.feature_weight_one'].copy_(f_weight_one)
#    state_dict['fuse.feature_weight_two'].copy_(f_weight_two)
#        
#def projection_weight_cal(element_list):
#    for j in range(0,len(element_list)):
#        value = element_list[j] + (1-sum(element_list[0:j+1]))/(j+1)
#        if value > 0:
#            index = j+1
#    lam = 1/(index)*(1-sum(element_list[0:index]))
#    for i in range(0 , len(element_list)):
#        element_list[i] = max(element_list[i]+lam,0)
#    return element_list
#
#def tensor_maker(T_1, T_2):
#    T_1 = T_1.t()
#    T_2 = T_2.t()
#    T_1 = T_1.cpu()
#    T_2 = T_2.cpu()
#    tensor_one = T_1.numpy()
#    tensor_two = T_2.numpy()
#    for i in range(0 , len(tensor_one)):
#        tensor_one_element = tensor_one[i]
#        tensor_two_element = tensor_two[i]
#        element_list = [tensor_one_element, tensor_two_element]
#        element_list.sort(reverse=True)
#        updated_weights = projection_weight_cal(element_list)
#        tensor_one[i] = updated_weights[0]
#        tensor_two[i] = updated_weights[1]
#    T_1 = T_1.t()
#    T_2 = T_2.t()
#    return (T_1, T_2)
        


def gpu_tensor(T1, T2, T3):
    BigT, _ = torch.cat([T1.t(), T2.t(), T3.t()], dim=1).sort()

    onesMask = torch.ones(BigT.shape).cuda()
    indexMask = torch.range(1, 3).repeat(BigT.shape[0], 1).cuda()
    acumMask = BigT.cumsum(1).cuda()

    ratioMask = (onesMask / indexMask)
    proportions = (1 - acumMask)
    averages = torch.mul(ratioMask, proportions)
    result = (BigT + averages)

    vmax, imax = torch.max((result > 0) * indexMask, axis=1)

    lam = (1/vmax) * (1 - acumMask[torch.arange(acumMask.shape[0]), imax])
    newlam = lam.reshape(lam.shape[0], 1)

    BigT = (BigT + newlam).clamp(min=0)

    return BigT

def projection_weight_cal(element_list):

    for j in range(0,len(element_list)):
        value = element_list[j] + (1-np.sum(element_list[0:j+1]))/(j+1)
        if value > 0:
            index = j+1

    lam = 1/(index)*(1-np.sum(element_list[0:index]))

    element_list = np.maximum(element_list+lam, 0)
    return element_list

def tensor_maker(T_1, T_2, T_3):
    T_1 = T_1.t()
    T_2 = T_2.t()
    T_3 = T_3.t()

    T_1 = T_1.cpu()
    T_2 = T_2.cpu()
    T_3 = T_3.cpu()

    tensor_one = T_1.numpy()
    tensor_two = T_2.numpy()
    tensor_three = T_3.numpy()

    for i in range(0 , len(tensor_one)):

        tensor_one_element = tensor_one[i]
        tensor_two_element = tensor_two[i]
        tensor_three_element = tensor_three[i]

        element_list = [tensor_one_element, tensor_two_element, tensor_three_element]
        element_list.sort(reverse=True)
        element_list = np.asarray(element_list)

        updated_weights = projection_weight_cal(element_list)

        tensor_one[i] = updated_weights[0]
        tensor_two[i] = updated_weights[1]
        tensor_three[i] = updated_weights[2]

    T_1 = T_1.t()
    T_2 = T_2.t()
    T_3 = T_3.t()

    return (T_1, T_2, T_3)
    
def matts_algorithm_gpu(model):

    state_dict = model.state_dict()


    fusion = ['fusionLayerOne', 'fusionLayerTwo']

    for i in fusion:

        t1_param = state_dict[i + '.feature_weight_one']
        t2_param = state_dict[i + '.feature_weight_two']
        t3_param = state_dict[i + '.feature_weight_three']

        f_weight_one = result[:,0]
        f_weight_two = result[:,1]
        f_weight_three = result[:,2]

        state_dict[i + '.feature_weight_one'].copy_(f_weight_one)
        state_dict[i + '.feature_weight_two'].copy_(f_weight_two)
        state_dict[i + '.feature_weight_three'].copy_(f_weight_three)
