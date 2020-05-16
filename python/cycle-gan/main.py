from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

from Losses.InitialLosses import ContentLoss, StyleLoss
from utils import loader, addModules, imgOptimizer, runStyleTransfer, imSave
from colourControl import loaderLum, recombine

# Launch the neural style transfer script
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Add arguments to the NST script.")
    parser.add_argument(
        '--numSteps',
        help="Number of steps of the script",
        type=int,
        default=300,
        required=False)
    parser.add_argument(
        '--styleWeight',
        help="Weight of the style",
        type=int,
        default=1000000)
    parser.add_argument(
        '--contentWeight',
        help="Weight of the content",
        type=int,
        default=10)
    parser.add_argument(
        '--styleImg',
        help="Style Image name",
        type=str,
        default="starry-night")
    parser.add_argument(
        '--contentImg',
        help="Content Image name",
        type=str,
        default="eiffel-tower")
    parser.add_argument(
        '--pretrained',
        help="Is the model pretrained",
        type=bool,
        choices=[True, False],
        default=True)
    parser.add_argument(
        '--model',
        help="Complete name of the model to use",
        type=str,
        default="vgg19")
    parser.add_argument(
        '--colourControl',
        help="Use colour control",
        type=bool,
        default=False)
    
    args = parser.parse_args()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dictionary mapping names of models to the model

    ## Hyperparameters
    imSize = 512 if torch.cuda.is_available() else 128
    numSteps = args.numSteps
    styleWeight = args.styleWeight
    contentWeight = args.contentWeight
    sLossesWeights = [1] * 5
    cLossesWeights = [1]

    
    normMean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normStd = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    # Layers over which the losses are computed
    contentLayers = ['conv_4']
    styleLayers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # Load the images
    st_name = args.styleImg
    ct_name = args.contentImg

    # Loads the model (only the features part, we don't need the classifier)
    # And put it in evaluation mode (!= training mode)
    modelName = args.model
    model = getattr(models, modelName)(pretrained=args.pretrained).features.to(device).eval()

    # If we use colour control
    if args.colourControl:
        rgbContent = loader("resources/style/"+st_name+".png", imSize, device)

        styleImg = loaderLum("resources/style/"+st_name+".png", imSize, device)
        contentImg = loaderLum("resources/content/"+ct_name+".png", imSize, device)
        inputImg = torch.randn(contentImg.data.size(), device=device)

        # Add our loss and normalization modules in the model
        styleModel, stLosses, ctLosses = addModules(model, normMean, normStd, styleImg, contentImg, contentLayers, styleLayers, device)

        # Run the algorithm
        output = runStyleTransfer(styleModel, inputImg, contentImg, styleImg, numSteps, styleWeight, contentWeight, stLosses, ctLosses, sLossesWeights, cLossesWeights)
        imSave(recombine(rgbContent, output), 'colourControl'+st_name+ct_name+modelName)

    else:
        styleImg = loader("resources/style/"+st_name+".png", imSize, device)
        contentImg = loader("resources/content/"+ct_name+".png", imSize, device)
        inputImg = torch.randn(contentImg.data.size(), device=device)

        # Add our loss and normalization modules in the model
        styleModel, stLosses, ctLosses = addModules(model, normMean, normStd, styleImg, contentImg, contentLayers, styleLayers, device)

        # Run the algorithm
        output = runStyleTransfer(styleModel, inputImg, contentImg, styleImg, numSteps, styleWeight, contentWeight, stLosses, ctLosses, sLossesWeights, cLossesWeights)
        imSave(output, st_name+ct_name+modelName)