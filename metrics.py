import json
import numpy as np
import pandas
import argparse
import sys


def thresold_metrics(predictions, labels, thresh=0.5):
    pred = predictions[:,1]
    lbl = labels[:,1]

    pred_thresholded = np.zeros_like(pred)
    indx = np.argwhere(pred>=thresh)
    pred_thresholded[indx] = 1

    true_positives = np.sum(pred_thresholded*labels[:,1])
    false_positives = np.sum(pred_thresholded * np.logical_not(labels[:,1]))
    false_negatives = np.sum(np.logical_not(pred_thresholded)*labels[:,1])

    precision = true_positives / (true_positives+false_positives)
    recall = true_positives / (true_positives+false_negatives)

    accuracy = np.average(np.logical_not(np.logical_xor(pred_thresholded, lbl)))

    #print(np.sum(pred_thresholded), np.sum(lbl))
    #return np.sum(pred_thresholded*lbl)/np.sum(lbl)

    return accuracy, precision, recall, true_positives, false_positives, false_negatives

def metrics_df(predictions, labels, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    metrics = ['accuracy', 'precision', 'recall', 'TP', 'FP', 'FN']
    data = []
    for thresh in thresholds:
        data.append(np.array(thresold_metrics(predictions, labels, thresh)))
    data = np.array(data)
    return pandas.DataFrame(data, thresholds, metrics)



if __name__ == '__main__':
    ## Parse input arguments from commandline
    aparser = argparse.ArgumentParser('Calculates Accuracy/Precision/Recall at thresholds [arange(0.1,0.9,0.2)] for given prediction/labels json dump file (produced by model_infer.py)')
    aparser.add_argument('--json-file', type=str, default='None', help='full path of the *inference.json file containing predictions/labels dictionary to load')

    args = aparser.parse_args()

    json_file = args.json_file
    if json_file == 'None':
        print('Inference file needed to calculate metrics...exiting.')
        sys.exit(1)

    with open(json_file, 'r') as f:
        inference_data = json.load(f)

    predictions = inference_data['predictions']
    labels = inference_data['labels']

    predictions = np.array(predictions)
    labels = np.array(labels)

    print(metrics_df(predictions, labels))
