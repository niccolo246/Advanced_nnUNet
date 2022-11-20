import torch
import torch.nn as nn
from torch import nn, cat


class InceptionA(nn.Module):

    def __init__(self, in_channels, out_channels, conv_block=None):
        super().__init__()

        if conv_block is None:
            conv_block = nn.Conv3d
        self.branch1x1 = conv_block(in_channels, int(out_channels/4), kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, int(out_channels/8), kernel_size=1)
        self.branch5x5_2 = conv_block(int(out_channels/8), int(out_channels/4), kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, int(out_channels/8), kernel_size=1)
        self.branch3x3dbl_2 = conv_block(int(out_channels/8), int(out_channels/4), kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(int(out_channels/4), int(out_channels/4), kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, int(out_channels/4), kernel_size=1)
        self.avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)  #
        self.final_conv = conv_block(out_channels, out_channels, kernel_size=3, stride=2)  #


    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avg_pool(x)  #
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return self.final_conv(torch.cat(outputs, 1))


##########################################################################


#input = torch.rand(1, 12, 12, 12, 12)

#layer = InceptionA(12, 32)
#layer = nn.Conv3d(12,32,kernel_size=3,stride=2,padding=1)

#output = layer(input)

#print(output.shape)












class BasicDenseBlock(nn.Module):
    def __init__(self, in_planes, growth_rate, kernel_size):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        self.conv1 = nn.Conv3d(in_planes, growth_rate, kernel_size, padding=1)


    def forward(self, x):

        out = self.conv1(x)

        return out

input = torch.rand(1, 12, 12, 12, 12)

layer = BasicDenseBlock(in_planes=12, growth_rate=32, kernel_size=3)

#layer = nn.Conv3d(12, 32, kernel_size=3, stride=2, padding=1)

output = layer(input)

print(output.shape)







class DenseLayer(nn.Module):
    """
    Here is the dense layer in which we have mutiple sub dense blocks contained (in truth I would call them layer containing the block
    although am sticking with nnUNet terminology)
    """
    def __init__(self, input_channels, output_channels, kernel_size, num_blocks,
                 block=BasicDenseBlock, growth_rate=10):
        super().__init__()


        self.convs = nn.ModuleList([block(input_channels + i*growth_rate, growth_rate, kernel_size)
                                    for i in range(num_blocks)])

        self.final_one_conv = nn.Sequential(nn.Conv3d(input_channels + (num_blocks)*growth_rate,
                                                                     output_channels, kernel_size=1))


        # this final conv will mean that the output is the desired number of channels
        self.final_pooling_conv = nn.Sequential(nn.Conv3d(output_channels, output_channels, kernel_size,
                                                                         padding=1, stride=1))

    def forward(self, x):

        for block in self.convs:
            out = block(x)
            x = cat([x, out], 1)  # 1 = channel axis
        return self.final_pooling_conv(self.final_one_conv(x))

input = torch.rand(1, 12, 12, 12, 12)

layer = DenseLayer(input_channels=12, output_channels=32, kernel_size=3, num_blocks=4)

#layer = nn.Conv3d(12, 32, kernel_size=3, stride=2, padding=1)

output = layer(input)

print(output.shape)










