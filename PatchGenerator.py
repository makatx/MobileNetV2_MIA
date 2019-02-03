from utils import getLabel, getWSI, getRegionFromSlide
import numpy as np
import openslide
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from itertools import cycle
import math

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
    detection_ratio = det_list_size / combined_list_size
    cls = dls + mls = dls + dls/sf = dls(1+1/sf) = dls (sf+1)/sf
    detection_ratio = dls/cls = dls/(dls * (sf+1)/sf) = sf/(sf+1)
    '''

    #detection_ratio = sample_factor / (sample_factor + 1)
    #true_batch = math.ceil(detection_ratio * batch_size)
    #all_batch_size = batch_size - true_batch

    #print('true_batch_size: {} \t all_batch_size: {}'.format(true_batch, all_batch_size))

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
                patch_img = getRegionFromSlide(getWSI(filename), level=level, start_coord=coords, dims=dims).astype(np.float)
                patch_img = (patch_img - 128) / 128
                patch.append(patch_img)

                ground_truth.append(getLabel(filename,level,coords,dims))

                #print('Level used: {}'.format(level))

            X_train = np.array(patch)

            if save_labels:
                labels_list.extend(ground_truth)
                yield X_train
            else:
                y_train = np.array(ground_truth)
                yield shuffle(X_train, y_train)
