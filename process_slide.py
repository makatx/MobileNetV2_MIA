import numpy as np
from utils import *
from ROIUtils import *
from ASAPAnnotationXML import *
from metrics import *
from keras.models import Model, load_model
import openslide
from math import ceil
from scipy.ndimage.measurements import label
from scipy.spatial.distance import pdist, squareform
import cv2
import argparse

def RunModelOnTiles(model, slide, tile_list, level=1, batch_size=64, dims=(256,256)):
    steps = ceil((tile_list.shape[0])/batch_size)
    gen = patch_batch_generator(slide, tile_list, batch_size, level=level, dims=dims)
    predictions = model.predict_generator(gen, steps, verbose=1)
    return predictions

def labelsList(filename, tile_list, level=1, dims=(256,256)):
    '''
    Return Labels for corresponding coordinates in tile_list, at given level and dimension tile for given file name
    '''
    labels = []
    for tile in tile_list:
        labels.append(getLabel(filename, level, tile, dims))
    return np.array(labels)

def markImage(slide, tile_list, predictions, return_mask=False, from_level=2, img_level=6, tile_size=256, threshold=0.4, color=[255,0,0]):
    if return_mask:
        img_shape = (slide.level_dimensions[img_level][1], slide.level_dimensions[img_level][0])
        img = np.zeros(img_shape, dtype=np.uint8)
        color = 255
    else:
        img = getRegionFromSlide(slide, level=img_level)
    scale_multiplier = slide.level_downsamples[from_level]/slide.level_downsamples[img_level]
    downsample = slide.level_downsamples[img_level]
    tile = ceil(tile_size * scale_multiplier)
    for i in range(len(tile_list)):
        if predictions[i,1] >= threshold:
            coord = ( int(tile_list[i][0]/downsample) , int(tile_list[i][1]/downsample) )
            img[coord[1]:coord[1]+tile, coord[0]:coord[0]+tile] = color
            #print('marking: ', coord, ' with tile: ', tile)
    return img

def getDetectedMaskList(mask):
    labels, detections = label(mask)
    mask_list = []
    for i in range(1, detections+1):
        m = np.zeros_like(mask)
        m[labels==i] = 255
        mask_list.append(m)
    return mask_list

def getBoundaryList(mask):
    mask_list = getDetectedMaskList(mask)
    boundary_list = []
    kernel = np.ones((2,2), dtype=np.uint8)
    for m in mask_list:
        grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
        boundary_list.append(grad)
    return boundary_list

def getBorderPointsList(mask):
    boundary_list = getBoundaryList(mask)
    borderpoints_list = []
    for b in boundary_list:
        borderpoints_list.append(np.argwhere(b))
    return borderpoints_list

def getAnnotation(borderpoints, coordinate_scale=1, downsample_rate=3):
    pairwise_distance = squareform(pdist(borderpoints))
    max_dist = np.max(pairwise_distance) + 100

    for i in range(pairwise_distance.shape[0]):
        pairwise_distance[i,i] = max_dist

    seq = [0]
    for i in range(pairwise_distance.shape[0]):
        seq.append(np.argmin(pairwise_distance[seq[-1]]))
        pairwise_distance[seq[-1], seq[-2]] = max_dist

    annotation = []
    for s in seq[::downsample_rate]:
        annotation.append((borderpoints[s]*coordinate_scale).astype(np.int64))

    return np.array(annotation)

def getAnnotationsList(mask, coordinate_scale=1, downsample_rate=3):
    borderpoints_list = getBorderPointsList(mask)

    annotation_list = []
    for b in borderpoints_list:
        annotation_list.append(getAnnotation(b, coordinate_scale, downsample_rate))

    return annotation_list

def evaluateModelOnSlide(modelfile, slide_files, level=1, tile_level= 7, batch_size=64, dims=(256,256)):
    '''
    Runs given model on given slide and returns dictionary containing 'predictions' and corresponding 'labels'

    parameters
    model: Keras model file to load and run on given slide
    slide_file: slide file to analyze
    level: level of the slide to run detection on
    tile_level: level to perform color dconvolution on and get tile list from
    batch_size: batch size to run model.predict_generator on
    dims: tile size
    '''
    #metrics = []
    results = {}
    model = load_model(modelfile)
    for slide_file in slide_files:
        slide = getWSI(slide_file)

        ## Get tile list using color deconvolution
        tile_list = getPatchCoordListFromFile('', slide_file, from_level=tile_level, with_filename=False)

        ## Run Model on tiles to get predictions
        predictions = RunModelOnTiles(model, slide, tile_list, level=level, batch_size=batch_size, dims=dims)

        ## Get labels list for the tile_list
        labels = labelsList(slide_file, tile_list, level=level, dims=dims)

        result[slide_file] = {'predictions': predictions.tolist(), 'labels': labels.tolist()}
        #metrics.append(metrics_df(predictions, labels))

    return results

if __name__ == '__main__':
    aparser = argparse.ArgumentParser('Accepts file with slide names to predict given model on and saves results to json file with labels')

    aparser.add_argument('--load-model', type=str, help='full path of the saved model file to load')
    aparser.add_argument('--slides-file', type=str, help='full path of the slides file to read names of files to read from')
    aparser.add_argument('--folder', type=str, default='', help='slides folder')
    aparser.add_argument('--out-file', type=str, default='preocess_slide_output.json',help='file name to save output to')
    aparser.add_argument('--print-metrics', action='store_true', help='Whether or not to output metrics info')

    args = aparser.parse_args()

    print('Got following arguments: \nload_model: {}\nslides_file: \
    {}\nout_file: {}\nprint_metric: {}'.format(args.load_model, args.slides_file, args.out_file, args.print_metrics))

    slide_files = []
    with open(args.slides_file, 'r') as f:
        for line in f.readlines():
            slide_files.append(args.folder+line.strip('\n'))
    results = evaluateModelOnSlide(args.load_model, slide_files, level=1, tile_level= 7, batch_size=128, dims=(256,256))

    with open(args.out_file, 'w') as f:
        json.dump(results, f)

    if args.print_metrics:
        print('\n\nPrinting Metrics:')
        for slide_file, res in results.items():
            print(slide_file)
            pred = np.array(res['predictions'])
            lbl = np.array(res['labels'])
            print(metrics_df(pred, lbl))
