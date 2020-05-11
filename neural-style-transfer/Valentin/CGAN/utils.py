import os
from PIL import Image
from random import randint
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Initializes the weights from N(0,0.02)
# Knowledge on how to initialize weights is taken from https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def init_weights(m):
    classname = m.__class__.__name__
    # For every convolutional layer in the network
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.02)
        m.bias.data.fill_(0)

# Since our dataset will contain two subsets, we need to create a new class
# that respects the API https://pytorch.org/docs/master/torchvision/datasets.html
class CGanDataset(torch.utils.data.Dataset):

    def __init__(self, transform, path1, path2):
        # path1 and path2 are the paths to the two subsets of images
        self.transform = transform
        self.imgs1 = [os.path.join(path1, i) for i in os.listdir(path1)]
        self.imgs2 = [os.path.join(path2, i) for i in os.listdir(path2)]

    def __getitem__(self, index):
        img1 = self.transform(Image.open(self.imgs1[index%len(self.imgs1)]))
        img2 = self.transform(Image.open(self.imgs2[index%len(self.imgs2)]))
        # No randomness, should we add it ?
        return {"A": img1, "B": img2}
    
    def __len__(self):
        return max(len(self.imgs1), len(self.imgs2))


# Loads the dataset for testing and training. 
# Must incorporate the A and B subsets.
def load_datasets(imSize, datasetName):
    transform = transforms.Compose([
        transforms.Resize([imSize, imSize]),
        transforms.ToTensor()
    ])
    return CGanDataset(transform, datasetName+'A', datasetName+'B')


def update_pool(pool, newImages, imgSize):
    # Updates the pool with the newImages and returns 
    # len(newImages) images.

    toReturn = []
    # Update the pool
    for image in newImages:
        if len(pool) < 50:
            pool.append(image)
        else:
            index = randint(0, 50-1)
            pool[index] = image
    
    # Fill the returned array
    for i in range(len(newImages)):
        index = randint(0, len(pool)-1)
        toReturn.append(pool[index].unsqueeze(0))

    x = torch.Tensor(size=[5]+list(newImages[0].size())).double()

    with torch.no_grad():
        torch.cat(toReturn, out=x)
        
    return x


class CustomLR:
    def __init__(self, nbEpochs, offset):
        self.nbEpochs = nbEpochs
        self.offset = offset
    
    def step(self, epoch):
        if 0 < epoch < self.offset:
            return 1
        else:
            return abs(epoch-self.nbEpochs) / (self.nbEpochs-self.offset)