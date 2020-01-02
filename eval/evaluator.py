import numpy as np


def r_at_k(prediction, true, k=10):
    all_recall = []
    for pred, t in zip(prediction, true):
        t = np.asarray(t)
        pred = np.asarray(pred)
        t_indices = np.argwhere(t)
        if t_indices.__len__() == 0:
            continue
        pred_indices = pred.argsort()[-k:][::-1] #sorting checkup
        pred_indices = list(pred_indices)
        for i in pred_indices:
            if i not in np.argwhere(pred):
                pred_indices.remove(i)

        recall = 0
        for t_index in t_indices:
            if t_index in pred_indices:
                recall += 1
        all_recall.append(recall / t_indices.__len__())
    return np.mean(all_recall), all_recall

def r_at_k_t2v(prediction, true, k=10):
    all_recall = []
    for pred, t in zip(prediction, true):
        t_indices = np.nonzero(t[0])[1]
        pred_indices = np.asarray(pred[:k])

        if t_indices.__len__() == 0:
            continue

        recall = 0
        for t_index in t_indices:
            if t_index in pred_indices:
                recall += 1
        all_recall.append(recall / t_indices.__len__())
    return np.mean(all_recall), all_recall


def p_at_k(prediction, true, k=10):
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
        all_precision.append(precision / pred_indices.__len__())
    return np.mean(all_precision), all_precision


def init_eval_holder(evaluation_k_set=None):
    if evaluation_k_set is None:
        evaluation_k_set = [10]

    dict = {}
    for k in evaluation_k_set:
        dict[k] = []
    return dict

