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
    def __init__(self, target):
        super(ContentLoss, self).__init__()

        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)

        return input


class StyleLoss(nn.Module):
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
