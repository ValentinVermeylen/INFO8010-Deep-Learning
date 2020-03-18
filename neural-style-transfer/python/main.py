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

# Partially inspired by : https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

###########
# Imports #
###########

import torch

import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from image import img_loader, img_show, img_export
from process import run_style_transfer


#####################
# General variables #
#####################

STYLE_IMG_PATH = '../resources/style/picasso.jpg'
CONTENT_IMG_PATH = '../resources/content/valentin.jpg'
RESULT_NAME = 'picasso-valentin'


########
# Main #
########

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###################
    # Load the images #
    ###################

    imsize = 512 if torch.cuda.is_available() else 128

    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])

    style_img = img_loader(STYLE_IMG_PATH, loader, device)
    content_img = img_loader(CONTENT_IMG_PATH, loader, device)

    assert style_img.size() == content_img.size(), "Error : need to import style and content images of the same size"

    # TODO : resize both images to the same size (instead of 'assert')

    #########################
    # Instantiate the model #
    #########################

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Reconvert into PIL image
    unloader = transforms.ToPILImage()

    plt.ion()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    ########################
    # Get the output image #
    ########################

    input_img = content_img.clone()
    output = run_style_transfer(cnn, device, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)

    ##############################
    # Display and export results #
    ##############################

    plt.figure()
    img_show(output, unloader, title='Output image')

    plt.ioff()
    plt.show()

    img_export(output, unloader, name=RESULT_NAME)
