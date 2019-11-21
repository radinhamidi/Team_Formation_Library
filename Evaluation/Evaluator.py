import numpy as np
from os import path
import json


def r_at_k(prediction, true, k=10):
    all_recall = []
    for pred, t in zip(prediction, true):
        t = np.asarray(t)
        pred = np.asarray(pred)

        t_indices = np.argwhere(t)
        if t_indices.__len__() == 0:
            continue
        pred_indices = pred.argsort()[-k:][::-1]

        recall = 0
        for t_index in t_indices:
            if t_index in pred_indices:
                recall += 1
        all_recall.append(recall/t_indices.__len__())
    return np.mean(all_recall), all_recall


def p_at_k(prediction, true, k=10): #Todo unit test of p@k and r@k
    all_precision = []
    for pred, t in zip(prediction, true):
        t = np.asarray(t)
        pred = np.asarray(pred)

        t_indices = np.argwhere(t)
        if t_indices.__len__() == 0:
            continue
        pred_indices = pred.argsort()[-k:][::-1]

        precision = 0
        for pred_index in pred_indices:
            if pred_index in t_indices:
                precision += 1
        all_precision.append(precision/pred_indices.__len__())
    return np.mean(all_precision), all_precision


def init_eval_holder(evaluation_k_set=None):
    if evaluation_k_set is None:
        evaluation_k_set = [10]

    dict = {}
    for k in evaluation_k_set:
        dict[k] = []
    return dict

def save_record(dict, dict_name, dir='../Output/'):
   json.dump(str(dict), open(dir+dict_name+'.json', 'w'))
