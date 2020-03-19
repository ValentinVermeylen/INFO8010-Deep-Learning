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

import copy

import torch.nn as nn

from loss import ContentLoss, StyleLoss
from image import get_input_optimizer, Normalization


#####################
# General variables #
#####################

CONTENT_LAYERS_DEFAULT = ['conv_4']
STYLE_LAYERS_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


#############
# Functions #
#############

def get_style_model_and_losses(cnn, device, norm_mean, norm_std, style_img, content_img, content_layers=CONTENT_LAYERS_DEFAULT, style_layers=STYLE_LAYERS_DEFAULT):
    cnn = copy.deepcopy(cnn)

    # Normalization module
    normalization = Normalization(norm_mean, norm_std).to(device)

    # Just in order to have an iterable access to or list of
    # content/syle losses
    content_losses = []
    style_losses = []

    # Assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    # Increment every time we see a conv
    i = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)

        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with
            # the ContentLoss and StyleLoss we insert below.
            # So we replace with out-of-place ones here.
            layer = nn.ReLU(inplace=False)

        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)

        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)

        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__)
            )

        model.add_module(name, layer)

        # Add content loss
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # Add style loss
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # We trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(cnn, device, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps, style_weight, content_weight):
    """
    Run the style transfer.
    """

    print('Building the style transfer model...')

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        device,
        normalization_mean,
        normalization_std,
        style_img,
        content_img
    )
    optimizer = get_input_optimizer(input_img)

    print('Optimizing...')
    print('')

    run = [0]
    style_scores, content_scores = list(), list()

    while run[0] < num_steps:
        def closure():
            # Correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            if run[0] % 20 == 0:
                print('[Run {}] STYLE LOSS : {:4f} - CONTENT LOSS : {:4f}'.format(run[0], style_score.item(), content_score.item()))

            style_scores.append(style_score.item())
            content_scores.append(content_score.item())

            return style_score + content_score

        optimizer.step(closure)

    print('')

    # Last correction
    input_img.data.clamp_(0, 1)

    return input_img, style_scores, content_scores
