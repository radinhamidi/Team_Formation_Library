import numpy as np
import ml_metrics as metrics
from keras_metrics.metrics import true_negative


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
        all_recall.append(recall / len(t_indices))
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


def find_indices(prediction ,true):
    preds = []
    trues = []
    for pred, t in zip(prediction, true):
        t = np.asarray(t)
        pred = np.asarray(pred)
        t_indices = np.argwhere(t)
        if t_indices.__len__() == 0:
            continue
        pred_indices = pred.argsort()[:][::-1] #sorting checkup
        pred_indices = list(pred_indices)
        for i in pred_indices:
            if i not in np.argwhere(pred):
                pred_indices.remove(i)
        preds.append(pred_indices)
        trues.append([int(t) for t in t_indices])
    return preds, trues

def mean_avg_precision_k_t2v(prediction ,true, k=10):
    preds = []
    trues = []
    for pred, t in zip(prediction, true):
        t_indices = np.nonzero(t[0])[1]
        pred_indices = np.asarray(pred[:k])

        if t_indices.__len__() == 0:
            continue
        preds.append(pred_indices)
        trues.append(t_indices)
    return metrics.mapk(trues, preds, k=k)

def init_eval_holder(evaluation_k_set=None):
    if evaluation_k_set is None:
        evaluation_k_set = [10]

    dict = {}
    for k in evaluation_k_set:
        dict[k] = []
    return dict

# #[u4, u1, u7, u2] va [u1, u2, u7]
# print(r_at_k([[0.3, 0.1, 0, 0.5, 0, 0, 0.2]], [[1,1,0,0,0,0,1]], k=1))#0/3 = 0
# print(r_at_k([[0.3, 0.1, 0, 0.5, 0, 0, 0.2]], [[1,1,0,0,0,0,1]], k=2))#1/3 = 0.33 ...
# print(r_at_k([[0.3, 0.1, 0, 0.5, 0, 0, 0.2]], [[1,1,0,0,0,0,1]], k=3))#2/3 = 0.66 ...
# print(r_at_k([[0.3, 0.1, 0, 0.5, 0, 0, 0.2]], [[1,1,0,0,0,0,1]], k=4))#1.0 = 1.0
#
# print(p_at_k([[0.3, 0.1, 0, 0.5, 0, 0, 0.2]], [[1,1,0,0,0,0,1]], k=1))#0/1
# print(p_at_k([[0.3, 0.1, 0, 0.5, 0, 0, 0.2]], [[1,1,0,0,0,0,1]], k=2))#1/2
# print(p_at_k([[0.3, 0.1, 0, 0.5, 0, 0, 0.2]], [[1,1,0,0,0,0,1]], k=3))#2/3
# print(p_at_k([[0.3, 0.1, 0, 0.5, 0, 0, 0.2]], [[1,1,0,0,0,0,1]], k=4))#3/4
#
# preds, trues = find_indices([[0.3, 0.1, 0, 0.5, 0, 0, 0.2]], [[1,1,0,0,0,0,1]])
# print(metrics.mapk(trues, preds, k=1))#0
# print(metrics.mapk(trues, preds, k=2))#0.25
# print(metrics.mapk(trues, preds, k=3))#0.388
# print(metrics.mapk(trues, preds, k=4))#0.638

