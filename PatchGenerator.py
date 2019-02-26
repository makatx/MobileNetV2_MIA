from utils import getLabel, getWSI, getRegionFromSlide
import numpy as np
import openslide
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from itertools import cycle
import math
from keras.preprocessing.image import ImageDataGenerator
import cv2

def patch_generator(folder, all_patch_list,
                    det_patch_list, batch_size=64,
                    sample_factor=1, levels=[0,1,2],
                    dims=(512,512),
                    save_labels=False, labels_list=None):
    '''
    Returns (via yields) the sample image patch and corresponding ground truth mask, in given batch_size, using
    one level in levels list per patch with equal probability

    if save_labels is True then the function appends the labels generated to labels_list and yields
    only the image patches - used for inference purposes (metrics)

    folder: location of the image slides

    sample_factor = det_list_size / mix_list_size
        if sample_factor = 1 then all patch sample size is same as detections patch sample size
        if sample_factor = 2 then all patch sample size is *half* as detections patch sample size
        if sample_factor = 0.5 then all patch sample size is *double* as detections patch sample size

    #detection_ratio = det_list_size / combined_list_size
    #cls = dls + mls = dls + dls/sf = dls(1+1/sf) = dls (sf+1)/sf
    #detection_ratio = dls/cls = dls/(dls * (sf+1)/sf) = sf/(sf+1)
    '''

    #detection_ratio = sample_factor / (sample_factor + 1)
    #true_batch = math.ceil(detection_ratio * batch_size)
    #all_batch_size = batch_size - true_batch

    #print('true_batch_size: {} \t all_batch_size: {}'.format(true_batch, all_batch_size))

    IDG = ImageDataGenerator()

    while 1:
        all_patch_list = shuffle(all_patch_list)
        det_patch_list = shuffle(det_patch_list)

        all_patch_list_size = math.ceil( len(det_patch_list) / sample_factor )
        all_patch_list_sub = all_patch_list[:all_patch_list_size]

        sampleset_size = len(all_patch_list_sub) + len(det_patch_list)

        ## Create a combined sample list and shuffle
        all_patch_list_sub.extend(det_patch_list)
        combined_sample_list = shuffle(all_patch_list_sub)

        for offset in range(0,sampleset_size,batch_size):

            sample_batch = combined_sample_list[offset:offset+batch_size]


            patch = []
            ground_truth = []

            for sample in sample_batch:
                filename = folder + sample[0]
                coords = sample[1]
                level = levels[np.random.randint(0, len(levels), dtype=np.int8)]

                brightness = (0.95 + np.random.rand()*(1.05-0.95)) if np.random.rand() > 0.5 else 1

                dims_factor = (0.9 + np.random.rand()*(1.-0.9)) if np.random.rand() > 0.5 else 1
                zoom_dims = int(dims[0] / dims_factor)

                flip_v = np.random.rand() > 0.5
                flip_h = np.random.rand() > 0.5

                transformation = {'brightness':brightness, 'flip_horizontal': flip_h, 'flip_vertical': flip_v}

                patch_img = getRegionFromSlide(getWSI(filename), level=level, start_coord=coords, dims=(zoom_dims, zoom_dims)).astype(np.float)
                if zoom_dims != dims[0]:
                    patch_img = cv2.resize(patch_img, dims)
                patch_img = IDG.apply_transform(patch_img, transformation)

                patch_img = (patch_img - 128) / 128
                patch.append(patch_img)

                ground_truth.append(getLabel(filename,level,coords,(zoom_dims, zoom_dims)))

                #print('Level used: {}'.format(level))

            X_train = np.array(patch)

            if save_labels:
                labels_list.extend(ground_truth)
                #print('||----------------------------------------------||')
                #print('len(patch): {}'.format(len(patch)))
                #print('len(ground_truth): {}'.format(len(ground_truth)))
                #print('len(labels_list): {}'.format(len(labels_list)))
                yield X_train
            else:
                y_train = np.array(ground_truth)
                yield shuffle(X_train, y_train)
