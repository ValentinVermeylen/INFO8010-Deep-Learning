import torch
import colorsys
from PIL import Image
import torchvision.transforms as transforms

def luminance(image):
    """
    Returns the luminance channel of the image.
    The formula for measuring colour luminance comes from
    https://stackoverflow.com/questions/6442118/python-measuring-pixel-brightness

    image is expected to be a torch tensor of size (3,N,N).
    Returns a NxN tensor.
    """
    # Compute the luminance for each pixel
    lum = torch.Tensor([0.2126, 0.7152, 0.0722])
    lum = lum.reshape(-1,1,1)

    ret = torch.sum(lum*image, dim=0)
    ret = ret.unsqueeze(0)
    ret = ret.unsqueeze(0)
    return ret
    
def loaderLum(imgName, imSize, device):
    """
    Loads the luminance of the image.
    Returns a imSizeximSize matrix of luminance.
    """

    ld = transforms.Compose([
        transforms.Resize([imSize, imSize]),
        transforms.ToTensor()
    ])

    img = Image.open(imgName)
    img = ld(img)
    img = img.to(device, torch.float)
    # If there are 4 channels (for example alpha channel of PNG images),
    # we discard it
    if img.size()[0] > 3:
        img = img[:3, :, :]
    
    return luminance(img)

def recombine(image, luminance):
    """
    Takes the image in RGB colorspace and the luminance and 
    returns the image in which the luminance has been applied
    using the HSL colorspace.
    """

    ret = torch.Tensor(3,image.size()[1], image.size()[2])
    # hls = colorsys.rgb_to_hls(image[0,:,:],image[1,:,:],image[2,:,:]) /255
    for size1 in range(image.size()[1]):
        for size2 in range(image.size()[2]):
            h,l,s = colorsys.rgb_to_hls(image[0,0,size1,size2]/255, image[0,1,size1,size2]/255, image[0,2,size1,size2]/255)
            l = luminance[size1, size2]
            r,g,b = colorsys.hls_to_rgb(h,l,s)
            ret[0, size1, size2] = r
            ret[1, size1, size2] = g
            ret[2, size1, size2] = b
    
    return ret

