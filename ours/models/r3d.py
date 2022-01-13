"""R3D"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)


        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.temporal_spatial_conv(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->groupnorm->ReLU->conv->groupnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride = 2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
            #self.downsamplegn  = nn.GroupNorm(num_groups = 16, num_channels = out_channels)

            # downsample with stride = 2 when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        #self.gn1 = nn.GroupNorm(num_groups = 16, num_channels = out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->groupnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        #self.gn2 = nn.GroupNorm(num_groups = 16, num_channels = out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x

class EncBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, layer_size, block_type):
        super(EncBlock, self).__init__()
        padding = (kernel_size-1)//2
        self.layers = nn.Sequential()
        self.layers.add_module('3DConv', nn.Conv3d(in_planes, out_planes, stride=1, padding=padding, kernel_size=kernel_size, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm3d(out_planes))
        #self.layers.add_module('GroupNorm', nn.GroupNorm(num_groups = 16, num_channels = out_planes))


    def forward(self, x):
        out = self.layers(x)
        return torch.cat([x,out], dim=1)
        #return out


class R3DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock, with_classifier=False, return_conv=False, num_classes=101, in_channels=3):
        super(R3DNet, self).__init__()
        self.with_classifier = with_classifier
        self.return_conv = return_conv
        self.num_classes = num_classes

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        # print("In_channels: ", in_channels)
        self.conv1 = SpatioTemporalConv(in_channels, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.bn1 = nn.BatchNorm3d(64)
        #self.gn1 = nn.GroupNorm(num_groups = 16, num_channels = 64)
        self.relu1 = nn.ReLU()
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # ENC BLOCK
        self.enc = EncBlock(in_planes=512, out_planes=512, kernel_size=1, layer_size=1, block_type=block_type)
        
        # if self.return_conv: #### COMMENTING THIS ####
        #     self.feature_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))   # 9216

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

        # if self.with_classifier: ### COMMENTING THIS AS WE do nit want to classify here ####
        #     self.linear = nn.Linear(512, self.num_classes)


    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
    
        # if self.return_conv:
        #     x = self.feature_pool(x)
        #     # print(x.shape)
        #     return x.view(x.shape[0], -1)
        
        # ENCODING
        enc_dim = 512 
        feat = self.enc(x)
        mu = feat[:,:enc_dim] 
        logvar = feat[:, enc_dim:]
        ##### USING first-half of conv5 output as mu and second-half as logvar for now
        # print("mu dim: ", mu.shape)
        # mu = x[:,:256] 
        # logvar = x[:,256:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print("std.shape: {}, mu.shape: {}".format(std.shape, mu.shape))
        feat = eps.mul(std * 0.001).add_(mu)

        # POOLING for passing to the FC layers in decoder
        x = feat
        x = self.pool(x)
        x = x.view(-1, enc_dim)

        # if self.with_classifier:
        #     x = self.linear(x)

        return x

class Regressor(nn.Module):
    def __init__(self, indim=1024, num_classes=5, in_channels=3): #indim = [orig_img_avg_pooled_features, trans_img_avg_pooled_features], num_classes is the number of possible transformations = 4}
        super(Regressor, self).__init__()

        fc1_outdim = 256

        # print("NUM_Classes: ", num_classes)
        self.r3d = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False, in_channels=in_channels)

        self.fc1 = nn.Linear(indim, fc1_outdim)
        self.fc2 = nn.Linear(fc1_outdim, num_classes)

        self.relu1 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
            
    def forward(self, x1, x2): # shape of x1 and x2 should be 5D = [BS, C=3, No. of images=clip_total_frames, 224, 224]
        x1 = self.r3d(x1)
        x2 = self.r3d(x2) # now the shape of x1 = x2 = BS X 512
        # print("x2 shae: ", x2.shape)
        x = torch.cat((x1,x2), dim=1)
        # print("x shape: ", x.shape)
        x = self.fc1(x)
        x = self.relu1(x)
        penul_feat = x
        x = self.fc2(penul_feat)

        return x

# if __name__ == '__main__':
#     r3d = R3DNet((1,1,1,1))