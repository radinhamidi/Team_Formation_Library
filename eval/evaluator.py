import numpy as np
import ml_metrics as metrics
from keras_metrics.metrics import true_negative

def r_at_k(prediction, true, k=10):
    all_recall = []
    for pred_indices, t_indices in zip(prediction, true):
        recall = 0
        for t_index in t_indices:
            if t_index in pred_indices[:k]:
                recall += 1
        all_recall.append(recall / len(t_indices))
    return np.mean(all_recall), all_recall


# def r_at_k_t2v(prediction, true, k=10, min_true=3):
#     all_recall = []
#     for pred, t in zip(prediction, true):
#         t_indices = np.nonzero(t[0])[1]
#         pred_indices = np.asarray(pred[:k])
#
#         if t_indices.__len__() == 0:
#             continue
#         if len(t_indices) < min_true:
#             continue
#         recall = 0
#         for t_index in t_indices:
#             if t_index in pred_indices:
#                 recall += 1
#         all_recall.append(recall / t_indices.__len__())
#     return np.mean(all_recall), all_recall


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


def find_indices(prediction, true, min_true=1):
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


def find_indices_t2v(prediction, true, min_true=1):
    preds = []
    trues = []
    for pred, t in zip(prediction, true):
        t_indices = np.nonzero(t[0])[1]
        pred_indices = list(np.asarray(pred))
        if t_indices.__len__() == 0:
            continue
        if len(pred_indices) == 0:
            pred_indices.append(-1)
        if len(t_indices) >= min_true:
            preds.append(pred_indices)
            trues.append([int(t) for t in t_indices])
    return preds, trues



def init_eval_holder(evaluation_k_set=None):
    if evaluation_k_set is None:
        evaluation_k_set = [10]

    dict = {}
    for k in evaluation_k_set:
        dict[k] = []
    return dict


def cal_relevance_score(prediction, truth):
    rs = []
    for p, t in zip(prediction, truth):
        r = []
        for p_record in p:
            if p_record in t:
                r.append(1)
            else:
                r.append(0)
        rs.append(r)
    return rs


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def coverage(pred, truth, k=10):
    pass


def help_hurt(pred_1, pred_2):
    len1 = len(pred_1)
    len2 = len(pred_2)
    min_len = min(len1, len2)
    if len1 != len2:
        print("Two given predictions have not same size.")

    diff = []
    for i,j in zip(pred_1[:min_len], pred_2[:min_len]):
        if (i-j)!=0:
            diff.append(i-j)
    diff = np.sort(diff)
    return diff


def team_formation_feasiblity(predictions, truth, user_skill_dict, k=10):
    score = 0
    for p, t in zip(predictions, truth):
        score += team_validtor(p, t, user_skill_dict, k)
    return score/len(predictions)


def team_validtor(p_users, t_users, user_skill_dict, k=10):
    having_skills = []
    required_skills = []

    for t_user in t_users:
        if t_user in user_skill_dict.keys():
            required_skills.extend(user_skill_dict[t_user])

    for p_user in p_users[:k]:
        if p_user in user_skill_dict.keys():
            having_skills.extend(user_skill_dict[p_user])

    for skill in required_skills:
        if skill not in having_skills:
            return 0
    return 1

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
