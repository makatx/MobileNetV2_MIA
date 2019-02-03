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

    args = aparser.parse_args()

    load_weights = args.load_weights
    sample_factor = args.sample_factor
    output_dir = args.output_dir
    out_prefix = args.out_prefix
    batch_size = args.batch_size

    print("Received following parameters:")
    print("batch_size={} \n load_weights={} \n output_dir={} \n sample_factor={} \n out_prefix={}".format(batch_size, load_weights, output_dir, sample_factor, out_prefix))


    with open('all_patch_list.json', 'rb') as f:
        all_patch_list = json.load(f)['list'][:20]

    with open('detections_patch_list.json', 'rb') as f:
        detections_patch_list = json.load(f)['list'][:20]

    dims = (256,256)
    input_patch = Input(shape=(dims[0],dims[1],3,))
    depth_multiplier = 0.5
    probs = MobileNetv2Classifier(input_patch, num_classes=2, output_stride=32, depth_multiplier=depth_multiplier)

    model = Model(input_patch, probs)
    labels_list = []
    generator = patch_generator('/home/mak/PathAI/slides/',
                                all_patch_list, detections_patch_list,
                                sample_factor=sample_factor,
                                batch_size=batch_size, dims=dims, levels=[1,2], save_labels=True, labels_list=labels_list)
    sampleset_size = math.ceil(len(detections_patch_list)/sample_factor) + len(detections_patch_list)
    steps = math.ceil(sampleset_size/batch_size)

    if load_weights != None:
        model.load_weights(load_weights)
        print('Loaded weights from {}'.format(load_weights))
    else:
        print('Weights needed to predict. Exiting.')
        sys.exit(1)

    predictions = model.predict_generator(generator, steps, verbose=1)

    output_data = {}
    output_data['predictions'] = predictions.tolist()
    output_data['labels'] = np.array(labels_list).tolist()

    print('Got {} predictions and {} labels\nsaving to json...'.format(predictions.shape[0], len(labels_list)))

    print('output_data[predictions] type: {}'.format(type(output_data['predictions'])))
    print('output_data[labels] type: {}'.format(type(output_data['labels'])))
    print(output_data)

    with open(output_dir+out_prefix+'inference.json', mode='w') as f:
        json.dump(output_data, f)
