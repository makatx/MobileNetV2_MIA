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
from datetime import datetime

if __name__ == '__main__':

    ## Parse input arguments from commandline
    aparser = argparse.ArgumentParser('Testing argument parser.')
    aparser.add_argument('--load-weights', type=str, help='full path of the checkpoint/model weights file to load')
    aparser.add_argument('--batch-size', type=int, default=32, help='batch_size to use')
    aparser.add_argument('--initial-epoch', type=int, default=0, help='starting epoch number to use for this run')
    aparser.add_argument('--epochs', type=int, default=1, help='number of epochs to run the training for')
    aparser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
    aparser.add_argument('--sample-factor', type=float, default=1, help='the ratio of true sample to all samples')
    aparser.add_argument('--checkpoint-dir', type=str, default='checkpoints/', help='location to store checkpoints/model weights after each epoch')
    aparser.add_argument('--log-dir', type=str, default='logs/', help='location to store fit.log (appended)')
    aparser.add_argument('--all-patch-list', type=str, help='full path of all_patch_list json file')
    aparser.add_argument('--detections-patch-list', type=str, help='full path of detections_patch_list json file')

    args = aparser.parse_args()

    batch_size = args.batch_size
    load_weights = args.load_weights
    initial_epoch = args.initial_epoch
    epochs = args.epochs
    learning_rate = args.learning_rate
    sample_factor = args.sample_factor
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    train_levels = [0,1]

    date = str(datetime.now().date())

    print("Received following parameters:")
    print("batch_size={} \n checkpoint_dir={} \n epochs={} \n initial_epoch={} \n learning_rate={} \
    \n load_weights={} \n log_dir={} \n sample_factor={}".format(batch_size, checkpoint_dir,
                                                                 epochs, initial_epoch, learning_rate,
                                                                 load_weights, log_dir, sample_factor))

    all_patch_list_json = args.all_patch_list if args.all_patch_list else 'all_patch_list.json'
    detections_patch_list_json = args.detections_patch_list if args.detections_patch_list else 'detections_patch_list.json'
    print('Using following patch list json files: \n{}\n{}'.format(all_patch_list_json, detections_patch_list_json))

    with open(all_patch_list_json, 'rb') as f:
        all_patch_list = json.load(f)['list']

    with open(detections_patch_list_json, 'rb') as f:
        detections_patch_list = json.load(f)['list']

    train_all_list, test_all_list = train_test_split(all_patch_list, test_size=0.1)
    train_true_list, test_true_list = train_test_split(detections_patch_list, test_size=0.1)

    dims = (256,256)
    input_patch = Input(shape=(dims[0],dims[1],3,))
    depth_multiplier = 0.5
    probs = MobileNetv2Classifier(input_patch, num_classes=2, output_stride=32, depth_multiplier=depth_multiplier)

    model = Model(input_patch, probs)

    train_generator = patch_generator('/home/mak/PathAI/slides/',
                                train_all_list, train_true_list,
                                sample_factor=sample_factor,
                                batch_size=batch_size, dims=dims, levels=train_levels)
    sampleset_size_train = math.ceil(len(train_true_list)/sample_factor) + len(train_true_list)
    steps_per_epoch = math.ceil(sampleset_size_train/batch_size)

    validn_generator = patch_generator('/home/mak/PathAI/slides/',
                                test_all_list, test_true_list,
                                sample_factor=sample_factor,
                                batch_size=batch_size, dims=dims, levels=train_levels)

    sampleset_size_validn = math.ceil(len(test_true_list)/sample_factor) + len(test_true_list)
    steps_per_epoch_validn = math.ceil(sampleset_size_validn/batch_size)

    checkpointer = ModelCheckpoint(checkpoint_dir+date+'_weights_imageAug_dropout_{epoch:02d}--{categorical_accuracy:.4f}--{val_loss:.4f}.hdf5', monitor='categorical_accuracy',
                               save_weights_only=True, save_best_only=True)
    csvlogger = CSVLogger(log_dir+'fit.log', append=True)

    if load_weights != None:
        model.load_weights(load_weights)
        print('Loaded weights from {}'.format(load_weights))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, callbacks=[checkpointer, csvlogger],
    validation_data=validn_generator, validation_steps=steps_per_epoch_validn, initial_epoch=initial_epoch)

    last_epoch = epochs
    model.save('modelsaves/'+date+'_mobilenetv2_model_camelyon17_imageAug_dropout_afterEpoch-'+str(last_epoch)+'.h5')
