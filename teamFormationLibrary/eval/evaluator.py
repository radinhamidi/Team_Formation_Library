import numpy as np


def find_indices(prediction, true, min_true=1):
    """Extracts the prediction and training indices
    ----------
    prediction : array-like, shape=(predicted_indices,)
        Extracts the predicted indices
    true : array-like, shape=(true_indices,)
        Extracts the true indices
    min_true : integer, optional (default=1)
        The minimum true valeus to be extracted
    """
    preds = []
    trues = []
    for pred, t in zip(prediction, true):
        t = np.asarray(t)
        pred = np.asarray(pred)
        t_indices = np.argwhere(t)
        if t_indices.__len__() == 0:
            continue
        pred_indices = pred.argsort()[:][::-1]  # sorting checkup
        pred_indices = list(pred_indices)
        pred_indices = [i for i in pred_indices if i in np.argwhere(pred)]
        if len(pred_indices) == 0:
            pred_indices.append(-1)
        if len(t_indices) >= min_true:
            preds.append(pred_indices)
            trues.append([int(t) for t in t_indices])
    return preds, trues


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
#
# testing relevance score computation
# print(cal_relevance_score([['u4', 'u1', 'u7', 'u2']],[['u1', 'u2', 'u7']])) #[[0, 1, 1, 1]]
# p, t = find_indices([[0.3, 0.1, 0, 0.5, 0, 0, 0.2]], [[1,1,0,0,0,0,1]])
# print(cal_relevance_score(p, t)) #[[0, 1, 1, 1]]
