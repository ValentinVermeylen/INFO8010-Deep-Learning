import torch
import torch.nn as nn

# Knowledge to create residual blocks taken from 
# https://github.com/trailingend/pytorch-residual-block/blob/master/main.py
class Residual(nn.Module):

    def __init__(self):
        super(Residual, self).__init__()

        self.resBlock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256,256,3),
            nn.InstanceNorm2d(256)
        )
    
    def forward(self, input_):
        return input_ + self.resBlock(input_)


class Generator(nn.Module):
    """Generator part of the network. Implemented from the appendix of
    https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, inputSize):
        # inputSize is a tuple containing the number of channels, 
        # the width and the height of the input image.

        super(Generator, self).__init__()
        _, nbChannels, width, height = inputSize

        self.model = nn.Sequential()

        # Add the first convolutional module
        self.model.add_module(nn.ReflectionPad2d(3))
        self.model.add_module(nn.Conv2d(nbChannels, 64, 7, stride=1))
        # Instance instead of batch, as stated in CycleGan paper
        self.model.add_module(nn.InstanceNorm2d(64))
        self.model.add_module(nn.ReLU(True))

        # Add the two next convolutional layers
        self.model.add_module(nn.Conv2d(64, 128, 3, stride=2, padding=1))
        self.model.add_module(nn.InstanceNorm2d(128))
        self.model.add_module(nn.ReLU(True))

        self.model.add_module(nn.Conv2d(128, 256, 3, stride=2, padding=1))
        self.model.add_module(nn.InstanceNorm2d(256))
        self.model.add_module(nn.ReLU(True))

        # Add the residual blocks
        nbBlocks = (6 if height == 128 else 9)
        for i in range(nbBlocks):
            self.model.add_module(Residual())

        # Add the half-strided convolutions
        self.model.add_module(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1))
        self.model.add_module(nn.InstanceNorm2d(128))
        self.model.add_module(nn.ReLU(True))

        self.model.add_module(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1))
        self.model.add_module(nn.InstanceNorm2d(64))
        self.model.add_module(nn.ReLU(True))

        # Add the last convolutional layer
        self.model.add_module(nn.ReflectionPad2d(3))
        self.model.add_module(nn.Conv2d(64, 3, 7, stride=1))
        self.model.add_module(nn.Tanh())
    
    def forward(self, input_):
        return self.model(input_)


# Implemented from https://arxiv.org/pdf/1703.10593.pdf
class Discriminator(nn.Module):
    def __init__(self):

        super(Discriminator, self).__init__()

        self.model = nn.Sequential()

        # Add the first block
        self.model.add_module(nn.Conv2d(3, 64, 4, stride=2, padding=1))
        self.model.add_module(nn.LeakyReLU(0.2))

        # Add the next 3 layers
        self.model.add_module(nn.Conv2d(64, 128, 4, stride=2, padding=1))
        self.model.add_module(nn.InstanceNorm2d(128))
        self.model.add_module(nn.LeakyReLU(0.2))

        self.model.add_module(nn.Conv2d(128, 256, 4, stride=2, padding=1))
        self.model.add_module(nn.InstanceNorm2d(256))
        self.model.add_module(nn.LeakyReLU(0.2))

        self.model.add_module(nn.Conv2d(256, 512, 4, stride=2, padding=1))
        self.model.add_module(nn.InstanceNorm2d(512))
        self.model.add_module(nn.LeakyReLU(0.2))

        # Add the next layer (the idea to include this one comes from https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
        # and has been confirmed by comparing our implementation with the one of the authors afterwards)
        self.model.add_module(nn.Conv2d(512, 512, 4, stride=1, padding=1))
        self.model.add_module(nn.InstanceNorm2d(512))
        self.model.add_module(nn.LeakyReLU(0.2))

        # Add the last layer to get a 1D output
        self.model.add_module(nn.Conv2d(512, 1, 4, stride=1, padding=1))

    def forward(self, input_):
        return self.model(input_)

# Do weight initialization