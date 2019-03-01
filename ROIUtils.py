import numpy as np, openslide
from utils import *
import cv2
from skimage.color import rgb2hed

def getThresholdMask(img, threshold=(140,210), channel=0, margins=None):
    '''
    Retuns threhold applied image for given threhold and channel, suppressing any pixels to 0 for given margins

    params:
    margins: (left_y, right_y, top_x, bottom_x) ;  can be specified as negative as well. ex: (50, -50, 50, -50)

    '''
    mask = np.zeros_like(img[:,:,channel], dtype=np.uint8)
    mask[((img[:,:,channel] > threshold[0]) & (img[:,:,channel] < threshold[1]))] = 255

    if margins != None :
        mask[:margins[0]] = 0
        mask[margins[1]:] = 0

        mask[:, :margins[2]] = 0
        mask[:, margins[3]:] = 0

    return mask


def getRedThresholdMask(img, threshold=(130,210), margins=(50, -50, 50, -50)):
    return getThresholdMask(img, threshold, channel=0, margins=margins)

def getGreenThresholdMask(img, threshold=(85,180), margins=(50, -50, 50, -50)):
    return getThresholdMask(img, threshold, channel=1, margins=margins)

def getBlueThresholdMask(img, threshold=(120,200), margins=(50, -50, 50, -50)):
    return getThresholdMask(img, threshold, channel=2, margins=margins)

def getHemaThresholdMask(hed_img, threshold=(30,100), margins=(50, -50, 50, -50)):
    return getThresholdMask(hed_img, threshold, channel=0, margins=margins)

def getEosinThresholdMask(hed_img, threshold=(120,200), margins=(50, -50, 50, -50)):
    return getThresholdMask(hed_img, threshold, channel=1, margins=margins)

def getDABThresholdMask(hed_img, threshold=(30,150), margins=(50, -50, 50, -50)):
    return getThresholdMask(hed_img, threshold, channel=2, margins=margins)

def performClose(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def performOpen(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def getHED(img):
    '''
    Return a channel scaled image in the HED, color deconvolution performed format
    '''
    hed = rgb2hed(img)
    #hed_sc = np.zeros_like(hed)
    for i in range(3):
        r_min = np.min(hed[:,:,i])
        r_max = np.max(hed[:,:,i])
        r = r_max - r_min
        hed[:,:,i] = (hed[:,:,i]-r_min) * 255.0/r
    return hed.astype(np.uint8)

def getMaskedImage(img, mask):
    masked = np.copy(img)
    masked[mask==0] = 0

    return masked

def blend(img1, img2, alpha):
    '''
    Adds the two images in weighted manner using corresponding weights
    '''
    return cv2.addWeighted(img1, alpha, img2, (1-alpha))


def getDABMask(img, margins=(25, -25, 25, -35)):
    '''
    Returns the bit mask for ROI from the given RGB img, using DAB channel of the HED converted image
    '''
    hed = getHED(img)
    mask = getDABThresholdMask(hed, margins=margins)
    mask = performOpen(performClose(mask))

    return mask

def getPatchCoordListFromFile(folder, filename, from_level='max', with_filename=False):
    path = folder+filename
    slide = getWSI(path)
    if from_level =='max':
        from_level = slide.level_count-1
    img = getRegionFromSlide(slide, level=from_level)
    mask = getDABMask(img)

    nzs = np.argwhere(mask)
    nzs = nzs * slide.level_downsamples[from_level]
    nzs = nzs.astype(np.int32)
    nzs = np.flip(nzs, 1)

    if not with_filename:
        return nzs

    l = []
    for i in range(nzs.shape[0]):
        l.append([filename, nzs[i].tolist()])

    return l

def getHighLevelDAB(folder, filename, from_level=7):
    path = folder+filename
    slide = getWSI(path)
    if from_level =='max':
        from_level = slide.level_count-1
    img = getRegionFromSlide(slide, level=from_level)
    return getDABMask(img)
