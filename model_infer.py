import numpy as np
import math
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Dense, Softmax
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger

from PatchGenerator import patch_generator
from mobilenetv2 import MobileNetv2Classifier


import json
import argparse
import sys


if __name__ == '__main__':

    ## Parse input arguments from commandline
    aparser = argparse.ArgumentParser('Runs Model on all patches and saves the predicitons and corresponding true labels for inference')
    aparser.add_argument('--load-weights', type=str, help='full path of the checkpoint/model weights file to load')
    aparser.add_argument('--sample-factor', type=float, default=1, help='the ratio of true sample to all samples')
    aparser.add_argument('--output-dir', type=str, default='inferences/', help='location to store the inference output files (prediction and corresponding labels)')
    aparser.add_argument('--out-prefix', type=str, default='output_', help='prefix for output files')
    aparser.add_argument('--batch-size', type=int, default=32, help='batch_size to use')
    aparser.add_argument('--all-patch-list', type=str, help='full path of all_patch_list json file')
    aparser.add_argument('--detections-patch-list', type=str, help='full path of detections_patch_list json file')
    aparser.add_argument('--depth-multiplier', type=float, default=0.75, help='depth multiplier for mobilenet')

    args = aparser.parse_args()

    load_weights = args.load_weights
    sample_factor = args.sample_factor
    output_dir = args.output_dir
    out_prefix = args.out_prefix
    batch_size = args.batch_size
    depth_multiplier = args.depth_multiplier

    print("Received following parameters:")
    print("batch_size={} \n load_weights={} \n output_dir={} \n sample_factor={} \n out_prefix={} \n depth_multiplier={}".format(batch_size, load_weights, output_dir, sample_factor, out_prefix, depth_multiplier))

    all_patch_list_json = args.all_patch_list if args.all_patch_list else 'all_patch_list.json'
    detections_patch_list_json = args.detections_patch_list if args.detections_patch_list else 'detections_patch_list.json'

    print('Using following patch list json files: \n{}\n{}'.format(all_patch_list_json, detections_patch_list_json))

    with open(all_patch_list_json, 'rb') as f:
        all_patch_list = json.load(f)['list']

    with open(detections_patch_list_json, 'rb') as f:
        detections_patch_list = json.load(f)['list']

    dims = (256,256)
    input_patch = Input(shape=(dims[0],dims[1],3,))
    probs = MobileNetv2Classifier(input_patch, num_classes=2, output_stride=32, depth_multiplier=depth_multiplier)

    model = Model(input_patch, probs)
    labels_list = []
    generator = patch_generator('/home/mak/PathAI/slides/',
                                all_patch_list, detections_patch_list,
                                sample_factor=sample_factor,
                                batch_size=batch_size, dims=dims, levels=[1], save_labels=True, labels_list=labels_list)
    sampleset_size = math.ceil(len(detections_patch_list)/sample_factor) + len(detections_patch_list)
    steps = math.ceil(sampleset_size/batch_size)

    #print('steps: {}'.format(steps))

    if load_weights != None:
        model.load_weights(load_weights)
        print('Loaded weights from {}'.format(load_weights))
    else:
        print('Weights file is needed to predict. Exiting.')
        sys.exit(1)

    predictions = model.predict_generator(generator, steps, verbose=1, workers=0)

    output_data = {}
    output_data['predictions'] = predictions.tolist()
    output_data['labels'] = np.array(labels_list).tolist()

    print('Got {} predictions and {} labels\nsaving to json...'.format(predictions.shape[0], len(labels_list)))

    with open(output_dir+out_prefix+'inference.json', mode='w') as f:
        json.dump(output_data, f)
