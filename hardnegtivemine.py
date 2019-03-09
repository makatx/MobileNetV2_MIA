import numpy as np
import json
from utils import *
from keras.models import Model, load_model
import openslide
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from math import ceil

def patch_batch_generator_from_jsonlist(folder, patch_list, batch_size=64, level=1, dims=(256,256)):
    '''
    Generating image batches from given json file (containing filename and coordinates) and level using coordinates in tile_list
    images are normalized: (x-128)/128 before being returned
    '''
    images = []
    b = 0
    for patch in patch_list:
        if b==batch_size:
            b=0
            images_batch = np.array(images)
            images = []
            yield images_batch
        slide  =getWSI(folder+patch[0])
        coord = patch[1]
        images.append(((getRegionFromSlide(slide, level, coord, dims=dims).astype(np.float))-128)/128)
        b +=1
    images_batch = np.array(images)
    yield images_batch

if __name__ == '__main__':

    ## Parse input arguments from commandline
    aparser = argparse.ArgumentParser('Save results from negative samples with labels for hard negative mining')
    aparser.add_argument('--load-model', type=str, default='modelsaves/2019-03-03_mobilenetv2_model_camelyon17_imageAug_dropout_trainLevel-0-01-01-01_afterEpoch-4.h5', help='full path of the checkpoint/model weights file to load')
    aparser.add_argument('--batch-size', type=int, default=128, help='batch_size to use')
    aparser.add_argument('--initial-sample', type=int, default=0, help='starting sampleset number to use for this run')
    aparser.add_argument('--level', type=int, default=1, help='level to run predictions on')
    aparser.add_argument('--out-file', type=str, default='hardnegative_out/out.json', help='location to store output')
    aparser.add_argument('--all-patch-list', type=str, default='all_patch_list_shuffled.json', help='full path of all_patch_list json file')

    args = aparser.parse_args()

    patches_file = args.all_patch_list
    print('Using patch file: ', patches_file)
    with open(patches_file, 'r') as f:
        all_patch_list = json.load(f)['list']

    #sample_size = 1251866
    sample_size = 64
    sampleset = args.initial_sample
    sampleset_start = sampleset*sample_size
    sampleset_end = (sampleset+1)*sample_size

    sample = all_patch_list[sampleset_start:sampleset_end]
    folder = '/home/mak/PathAI/slides/'
    batch_size = args.batch_size
    level = args.level
    dims = (256,256)
    model_file = args.load_model
    gen = patch_batch_generator_from_jsonlist(folder, sample, batch_size, level, dims)

    model = load_model(model_file)
    predictions = model.predict_generator(gen, ceil(sample_size/batch_size), verbose=1)

    labels = []
    for s in sample:
        labels.append(getLabel(folder+s[0], level, s[1], dims))
    labels = np.array(labels)

    data = {'sample':sample, 'predictions':predictions.tolist(), 'labels':labels.tolist()}

    with open(args.out_file, 'w') as f:
        json.dump(data, f)
