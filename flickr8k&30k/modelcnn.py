import torch
import torchvision.models as models
from torch import nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CNN(nn.Module):
    def __init__(self, encodingsize = 14):
        super(CNN, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        layerlist = list(vgg16.children())
        #print(*layerlist)
        self.vgg16 = nn.Sequential(*layerlist[0][:-1]) #removed all pool including maxpool
        self.pool_reshape = nn.AdaptiveAvgPool2d((encodingsize,encodingsize))
        for parameter in self.vgg16.parameters():
            parameter.requires_grad = False #TODO:check correctness/fine tune
    def forward(self, input):
        #input dimension: batchsize*3*224*224
        x = self.vgg16(input)
        # print('1. ', x.shape)
        #input dimension: batchsize*512*14*14 (inputH/16)
        x = self.pool_reshape(x)
        # print('2. ', x.shape)
        #input dimension: batchsize*512*14*14 (inputH/16)
        x = x.permute(0, 2, 3, 1)
        # print('3. ', x.shape)
        #output dimension: batchsize*14*14*512 #TODO:check
        return x
