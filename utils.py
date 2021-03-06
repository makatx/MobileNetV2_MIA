import openslide
import numpy as np
from PIL import Image
import xml.etree.cElementTree as ET
import cv2
import matplotlib.pyplot as plt
import os

class Annotation:

    scaleFactor = 1
    coords_orig = []
    coords_order = []
    coords_list = []
    bounds = []
    bounds_orig = []

    def __init__(self, filename, scaleFactor=1):
        self.scaleFactor = scaleFactor
        with open(filename, 'rb') as f:
            self.root = ET.parse(f)
        self.coords_orig = []
        self.coords_order = []
        self.group = []
        self.type = []

        for annot in self.root.iter('Annotation'):
            coords_tag = annot.find('Coordinates')
            lst = []
            for coord in coords_tag.findall('Coordinate'):
                lst.append([float(coord.attrib['Order']), float(coord.attrib['X']), float(coord.attrib['Y'])])
            n = np.array(lst)
            n = n[n[:,0].argsort()]
            self.coords_orig.append(n[:,1:])
            self.coords_order.append(n)
            self.group.append(annot.attrib['PartOfGroup'])
            self.type.append(annot.attrib['Type'])

        self.coords_list = self.scale(factor=scaleFactor)
        self.calcBounds()

    def scale(self, coords=None, factor=1):
        if coords == None: coords = self.coords_orig
        coords_scaled = []
        for n in range(len(coords)):
            coords_scaled.append((coords[n] / factor).astype(np.int));
        return coords_scaled

    def shift(self, coords=None, origin=(0,0)):
        if coords == None: coords = self.coords_orig
        shifted = []
        origin = np.array(origin)
        for n in coords:
            shifted.append(n - origin)
        return shifted

    def calcBounds(self):
        bounds = []
        for n in self.coords_list:
            xmin = n[:,0].min()
            ymin = n[:,1].min()
            xmax = n[:,0].max()
            ymax = n[:,1].max()
            bounds.append(np.array([xmin,ymin,xmax,ymax]))
        self.bounds = np.array(bounds)
        bounds = []
        for n in self.coords_orig:
            xmin = n[:,0].min()
            ymin = n[:,1].min()
            xmax = n[:,0].max()
            ymax = n[:,1].max()
            bounds.append(np.array([xmin,ymin,xmax,ymax]))
        self.bounds_orig = np.array(bounds)


def getWSI(filename):
    '''
        Returns image for desired level from given OpenSlide WSI format image filename

    '''
    slide = openslide.OpenSlide(filename)

    return slide

def getRegionFromSlide(slide, level=8, start_coord=(0,0), dims='full', from_level=8):
    if dims == 'full':
        img = np.array(slide.read_region((0,0), level, slide.level_dimensions[level]))
        img = img[:,:,:3]
    else:
        img = np.array(slide.read_region(start_coord, level, dims ))
        img = img[:,:,:3]

    return img

def getGTmask(img_filename, annotn_filename, level, coords, dims):
    slide = getWSI(img_filename)
    ann = Annotation(annotn_filename)
    c_shifted = ann.shift(origin=coords)
    c_scaled = ann.scale(c_shifted, slide.level_downsamples[level])

    mask = cv2.fillPoly(np.zeros((dims[0],dims[1],1)), c_scaled, (1))

    return mask

def getLabel(filename, level, coords, dims):
    '''
    Check if the annotation file with same name (extension .xml) exists: if not, return all zero mask of shape (dims,1)
    else, get the annotation file, shift its coordinates by coords and scale using level in slide downsample,
    followed by polyFill operation on a all zero mask of dimension (dims,1) with 1 and return it
    '''
    annotn_filename, _ = os.path.splitext(filename)
    annotn_filename = annotn_filename + '.xml'

    if os.path.exists(annotn_filename):
        detection = np.any(getGTmask(filename, annotn_filename, level, coords, dims))
        label = np.array( [float(not detection), float(detection)] )
        #print('label: {}'.format(label))
        return label
    else:
        #print('{} does not exist'.format(annotn_filename))
        return np.array([1.0, 0.0])

def getTileList(slide, level=2, tile_size=256):
    '''
    Returns a list of coordinates for starting point of tile of given tile_size square for the given slide at given level
    converted/scaled to full dimension of slide
    '''
    dims = slide.level_dimensions[level]
    tile_list = []
    width = dims[0]-tile_size
    height = dims[1]-tile_size

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile_list.append([x,y])
        tile_list.append([width,y])
    for x in range(0, width, tile_size):
        tile_list.append([x, height])
    tile_list.append([width, height])

    tile_list = (np.array(tile_list) * slide.level_downsamples[level]).astype(np.int32)

    return tile_list

def patch_batch_generator(slide, tile_list, batch_size, level=2, dims=(256,256)):
    '''
    Generating image batches from given slide and level using coordinates in tile_list
    images are normalized: (x-128)/128 before being returned
    '''
    images = []
    b = 0
    for coord in tile_list:
        if b==batch_size:
            b=0
            images_batch = np.array(images)
            images = []
            yield images_batch
        images.append(((getRegionFromSlide(slide, level, coord, dims=dims).astype(np.float))-128)/128)
        b +=1
    images_batch = np.array(images)
    yield images_batch

'''
Test code:
mask = getLabel( 'patient_015/patient_015_node_2.tif', 2,  [ 67700, 101508], (512,512))
print('Mask sum: {} ; shape: {}'.format(np.sum(mask), mask.shape))
plt.imshow(np.reshape(mask, (512,512)))
'''
