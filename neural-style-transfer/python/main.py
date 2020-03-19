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

# Partially inspired by
# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

###########
# Imports #
###########

import os

import torch

import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from image import img_loader, img_show, img_export
from process import run_style_transfer
from chart import line_chart


#####################
# General variables #
#####################

STYLE_PATH = '../resources/style/'
CONTENT_PATH = '../resources/content/'
RESULT_PATH = '../results/'

STYLE_IMG = 'starry-night.png'
CONTENT_IMG = 'eiffel-tower.png'
RESULT_IMG = 'starry-night-eiffel-tower.png'

NUM_STEPS = 300
STYLE_WEIGHT = 1000000
CONTENT_WEIGHT = 10


########
# Main #
########

if __name__ == '__main__':
    ##################
    # Initialisation #
    ##################

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Display information
    print('#################################################')
    print('# NEURAL STYLE TRANSFER - Deep learning project #')
    print('#################################################')
    print('')

    ###################
    # Load the images #
    ###################

    # Set the image size
    imsize = 512 if torch.cuda.is_available() else 128

    # Create the loader
    loader = transforms.Compose([
        transforms.Resize([imsize, imsize]),
        transforms.ToTensor()
    ])

    # Load the images
    style_img = img_loader(STYLE_PATH + STYLE_IMG, loader, device)
    content_img = img_loader(CONTENT_PATH + CONTENT_IMG, loader, device)

    # Display information
    print('Load the images')
    print('---------------')
    print('Style image : {} ({})'.format(STYLE_IMG, style_img.size()))
    print('Content image : {} ({})'.format(CONTENT_IMG, content_img.size()))
    print('')

    #########################
    # Instantiate the model #
    #########################

    # Instantiate the VGG19 model
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # VGG networks are trained on images with each channel normalized
    # by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # We will use them to normalize the image before sending it into
    # the network.
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    ########################
    # Get the output image #
    ########################

    # Display information
    print('Parameters')
    print('----------')
    print('Number of steps : {}'.format(NUM_STEPS))
    print('Style weight : {}'.format(STYLE_WEIGHT))
    print('Content weight : {}'.format(CONTENT_WEIGHT))
    print('')
    print('Get the output image')
    print('--------------------')

    # Get the output image
    input_img = content_img.clone()
    output, style_scores, content_scores = run_style_transfer(
        cnn,
        device,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img, style_img,
        input_img,
        NUM_STEPS,
        STYLE_WEIGHT,
        CONTENT_WEIGHT
    )

    #####################
    # Export the result #
    #####################

    # Create the unloader (to reconvert into PIL image)
    unloader = transforms.ToPILImage()

    # Export the output image
    img_export(output, unloader, name=RESULT_PATH + RESULT_IMG)

    # Export the line chart
    fig_name = '{}-{}-{}-{}.{}'.format(
        RESULT_PATH + os.path.splitext(RESULT_IMG)[0],
        NUM_STEPS,
        STYLE_WEIGHT,
        CONTENT_WEIGHT,
        'pdf'
    )
    line_chart(
        list(range(1, len(style_scores) + 1)),
        [style_scores, content_scores],
        'Number of steps',
        'Loss',
        ['Style', 'Content'],
        fig_name
    )

    # Display information
    print('Job done !')
    print('Output image saved as {}'.format(RESULT_PATH + RESULT_IMG))
    print('Line chart saved as {}'.format(fig_name))
