"""
INFO8010-1 - Deep learning

University of Liege
Academic year 2019-2020

Project : Neural style transfer

Authors :
    - Maxime Meurisse
    - Adrien Schoffeniels
    - Valentin Vermeylen
"""

###########
# Imports #
###########

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from PIL import Image


#############
# Functions #
#############

def img_loader(img_name, loader, device):
    img = Image.open(img_name)
    img = loader(img).unsqueeze(0)
    img = img.to(device, torch.float)

    # If there are 4 channels (for example alpha channel of PNG images),
    # we discard it
    if img.size()[1] > 3:
        img = img[:, :3, :, :]

    return img


def img_show(tensor, unloader, title=None):
    # We clone the tensor to not do changes on it
    image = tensor.cpu().clone()

    # We remove the fake batch dimension
    image = image.squeeze(0)
    image = unloader(image)

    plt.imshow(image)

    if title is not None:
        plt.title(title)

    # We pause a bit so that plots are updated
    plt.pause(0.001)


def img_export(tensor, unloader, name):
    # We clone the tensor to not do changes on it
    image = tensor.cpu().clone()

    # We remove the fake batch dimension
    image = image.squeeze(0)
    image = unloader(image)

    image.save(name)


def get_input_optimizer(input_img):
    # We show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    return optimizer


###########
# Classes #
###########

class Normalization(nn.Module):
    """
    Module to normalize input image so we can easily put it in a nn.Sequential.
    """

    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
