import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Architecture from https://www.kaggle.com/keras/vgg19
class CNNModel(nn.Module):
    def __init__(self):  
        super(CNNModel, self).__init__()  
        self.conv1 = nn.Conv2d(3, 64, (3, 3))
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 64, (3,3))
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv3 = nn.Conv2d(3, 128, (3, 3))
        self.conv4 = nn.Conv2d(3, 128, (3, 3))

        self.conv5 = nn.Conv2d(3, 256, (3, 3))
        self.conv6 = nn.Conv2d(3, 256, (3, 3))
        self.conv7 = nn.Conv2d(3, 256, (3, 3))
        self.conv8 = nn.Conv2d(3, 256, (3, 3))

        self.conv9 = nn.Conv2d(3, 512, (3, 3))
        self.conv10 = nn.Conv2d(3, 512, (3, 3))
        self.conv11 = nn.Conv2d(3, 512, (3, 3))
        self.conv12 = nn.Conv2d(3, 512, (3, 3))

        self.conv13 = nn.Conv2d(3, 512, (3, 3))
        self.conv14 = nn.Conv2d(3, 512, (3, 3))
        self.conv15 = nn.Conv2d(3, 512, (3, 3))
        self.conv16 = nn.Conv2d(3, 512, (3, 3))

    def forward(self, x):

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.act(out)
        out = self.conv4(x)
        out = self.act(out)
        out = self.pool(out)

        out = self.conv5(out)
        out = self.act(out)
        out = self.conv6(x)
        out = self.act(out)
        out = self.conv7(out)
        out = self.act(out)
        out = self.conv8(x)
        out = self.act(out)
        out = self.pool(out)

        out = self.conv9(out)
        out = self.act(out)
        out = self.conv10(x)
        out = self.act(out)
        out = self.conv11(out)
        out = self.act(out)
        out = self.conv12(x)
        out = self.act(out)
        out = self.pool(out)

        out = self.conv13(out)
        out = self.act(out)
        out = self.conv14(x)
        out = self.act(out)
        out = self.conv15(out)
        out = self.act(out)
        out = self.conv16(x)
        out = self.act(out)
        out = self.pool(out)

        return out
    

# Content loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# Style loss
def gram_matrix(input):
    # batch size, nb of feature maps, (dimension of feature map)
    a, b, c, d = input.size()

    features = input.view(a*b, c*d)

    G = torch.mm(features, features.t())

    return G.div(a*b*c*d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input