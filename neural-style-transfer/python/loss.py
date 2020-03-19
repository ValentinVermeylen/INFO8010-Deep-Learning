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
import torch.nn.functional as F


###########
# Classes #
###########

class ContentLoss(nn.Module):
    """
    The content loss is a function that represents a weighted
    version of the content distance for an individual layer.

    The function takes the feature maps F_{XL} of a layer L in
    a network processing input X and returns the weighted content
    distance w_{CL}.D^{L}_{C}(X,C) between the image X and the
    content image C.
    """

    def __init__(self, target):
        super(ContentLoss, self).__init__()

        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)

        return input


class StyleLoss(nn.Module):
    """
    The style loss module is implemented similarly to the content
    loss module. It will act as a transparent layer in a network
    that computes the style loss of that layer.
    """

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()

        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)

        return input
