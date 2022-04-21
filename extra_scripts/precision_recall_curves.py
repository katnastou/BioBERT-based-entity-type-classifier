#!/usr/bin/python
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
# from matplotlib import pyplot
import csv
import numpy as np
from argparse import ArgumentParser
import sys
from sklearn.metrics import roc_curve
#np.seterr(divide='ignore', invalid='ignore')


def argparser():
    ap = ArgumentParser()
    ap.add_argument('predictionfile', help=' TSV file with labels')
    ap.add_argument('probsfile', help=' TSV file with probabilities')    
    return ap

#from my prediction file where the predicted label is appended in the end
def get_labels_and_predictions(input_file, quotechar=None):
    y_true = []
    y_pred = []
    with open(input_file, "r") as f:
        lines = csv.reader(f, delimiter="\t",quotechar=quotechar)
        for line in lines:
            true_label = line[9]
            prediction = line[10]
            if true_label=="Complex_formation":
                y_true.append(1)
            else:
                y_true.append(0)
            if prediction=="Complex_formation":
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_true,y_pred

def get_probabilities(input_file, quotechar=None):
    negative_probs = []
    positive_probs = []
    with open(input_file, "r") as f:
        lines = csv.reader(f, delimiter="\t",quotechar=quotechar)
        for line in lines:
            negative_prob = line[0]
            positive_prob = line[1]
            negative_probs.append(np.float32(negative_prob))
            positive_probs.append(np.float32(positive_prob))
        return negative_probs,positive_probs

def main(argv):
    args = argparser().parse_args(argv[1:])

    #true and predict labels for test set
    y_true,y_pred = get_labels_and_predictions(args.predictionfile)
    #convert to numpy arrays
    testy=np.array(y_true)
    yhat=np.array(y_pred)

    neg_probs,pos_probs = get_probabilities(args.probsfile)
    positive_probs = np.array(pos_probs)
    precision, recall, thresholds = precision_recall_curve(testy, positive_probs)

    aucprrec = auc(recall, precision)

    f1_scores = 2*recall*precision/(recall+precision)

    f1_scores = f1_scores[np.logical_not(np.isnan(f1_scores))]
    # print (f1_scores)
    print('Best threshold: ', thresholds[np.argmax(f1_scores)])
    print('Best F1-Score: ', np.max(f1_scores))
    print("Area Under Precision Recall Curve={}".format(aucprrec))
    print("{:.2%}\t{:.2%}".format(np.max(f1_scores),aucprrec))
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
