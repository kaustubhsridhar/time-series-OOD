import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Prev Enc blcok with Conv layer
# class EncBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(EncBlock, self).__init__()
#         self.layers = nn.Sequential()
#         self.layers.add_module('2DConv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))

#     def forward(self, x):
#         out = self.layers(x)
#         return torch.cat([x,out], dim=1)
#         #return out

# New Enc Block with Linear Layer
class EncBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(EncBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Linear', nn.Linear(in_features=in_features, out_features=out_features))

    def forward(self, x):
        out = self.layers(x)
        return torch.cat([x,out], dim=1)
        #return out

class LeNet5(nn.Module):

    def __init__(self, n_classes, in_feat=1080):
        super(LeNet5, self).__init__()
        
        self.layer1 = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding='same'),
            nn.Tanh())
        
        self.layer2 = nn.AvgPool2d(kernel_size=2)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.Tanh())
        
        self.layer4 = nn.AvgPool2d(kernel_size=2)

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1, padding='same'),
            nn.Tanh())

        # prev enc block
        # self.enc = EncBlock(in_channels=120, out_channels=120, kernel_size=5, stride=1)

        self.layer6 =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_feat, out_features=84), # in_features 1440 for cl 18/20, 1080 for cl 16
            nn.Tanh())

        # new enc block
        self.enc = EncBlock(in_features=84, out_features=84)

        # no classification required here
        # self.layer7 =  nn.Linear(in_features=84, out_features=n_classes) 

        self.all_layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6 ]#, self.layer7]


    def forward(self, x):
        x = self.layer1(x) #Conv2D
        x = self.layer2(x) #AvgPool
        x = self.layer3(x) #Conv2D
        x = self.layer4(x) #AvgPool
        x = self.layer5(x) #Conv2D
        # print("x shape:", x.shape)
        x = self.layer6(x) #Linear
        # print("x shape after flatten: ", x.shape)
        # logits = self.layer7(x) #Linear
        
        # ENCODING
        enc_dim = 84 
        feat = self.enc(x)
        mu = feat[:,:enc_dim] 
        logvar = feat[:, enc_dim:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # print("std.shape: {}, mu.shape: {}".format(std.shape, mu.shape))
        feat = eps.mul(std * 0.001).add_(mu)

        return feat # x
    
class Regressor(nn.Module):
    def __init__(self, indim=168, num_classes=4, wl=16): #indim = [orig_img_avg_pooled_features, trans_img_avg_pooled_features] = 2*enc_dim from forward, num_classes is the number of possible transformations = 4}
        super(Regressor, self).__init__()

        self.num_classes = num_classes

        fc1_outdim = 42

        if wl == 16:
            in_feat = 1080
        else:
            in_feat = 1440
        self.lenet = LeNet5(n_classes = self.num_classes, in_feat=in_feat)

        self.fc1 = nn.Linear(indim, fc1_outdim)
        self.fc2 = nn.Linear(fc1_outdim, num_classes)

        self.relu1 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
            
    def forward(self, x1, x2): # shape of x1 and x2 should be 5D = [BS, C=3, No. of images=clip_total_frames, 224, 224]
        # print("x2 shape before: ", x2.shape)
        x1 = self.lenet(x1)
        x2 = self.lenet(x2) # now the shape of x1 = x2 = BS X 512
        # print("x2 shape: ", x2.shape)
        x = torch.cat((x1,x2), dim=1)
        # print("x shape: ", x.shape)
        x = self.fc1(x)
        # print("After fc1, x shape:  ", x.shape)
        x = self.relu1(x)
        penul_feat = x
        x = self.fc2(penul_feat)
        # print(x)

        return x
