import csv
import pickle as pkl
from dal.load_dblp_data import *
from cmn.utils import crossValidate
import eval.evaluator as dblp_eval
import eval.ranking as rk
from eval import plotter
import ml_metrics as metrics
import dal.load_dblp_data as dblp


seed = 7
np.random.seed(seed)
k = 50
# year = 2009
year = 2017


# fax = './x_sampleset.pkl'
# fay = './y_sampleset.pkl'



# Variables
# train_ratio = 0.8
# validation_ratio = 0.0
# epochs = 30
# batch_sizes = 8
# sp = 0.01
# b_val = 3  # Controls the acitvity of the hidden layer nodes
# encoding_dim = 3000
#
## Get the train and test as done in DL assignments.
## Train/Validation/Test version
# x_train, x_validate, x_test, ids = crossValidate(x, train_ratio, validation_ratio)
# y_train = y[ids[0:int(y.__len__() * train_ratio)]]
# y_validate = y[ids[int(y.__len__() * train_ratio):int(y.__len__() * (train_ratio + validation_ratio))]]
# y_test = y[ids[int(y.__len__() * (train_ratio + validation_ratio)):]]
#
# input_dim = x_train.shape[1]
# output_dim = y_train.shape[1]
# print("Input/output dimensions ", input_dim, output_dim)



### Writing target teams skills into file
# train_test_indices = dblp.load_train_test_indices()
# dataset = dblp.load_preprocessed_dataset()
# fold_counter = 1
# x_train, y_train, x_test, y_test = dblp.get_fold_data(fold_counter, dataset, train_test_indices)
# skill_dir='../dataset/invertedTermCount.txt'
# skills, skills_freq = load_skills(skill_dir)
# skills = np.asarray(skills)
#
# with open('test_skills.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for skill_record in x_test:
#         skill_set = []
#         for skill_id in np.nonzero(skill_record)[0]:
#             skill_set.append(skills[skill_id])
#         writer.writerow(skill_set)



##### After running baseline code
def calc_r_at_k(prediction, true):
    all_recall = []
    for pred, t in zip(prediction, true):
        recall = 0
        for t_record in t:
            if t_record in pred:
                recall += 1
        all_recall.append(recall/t.__len__())
    return np.mean(all_recall), all_recall


def filter_len(prediction ,true, min_true=1):
    preds = []
    trues = []
    for p, t in zip(prediction, true):
        if len(t) >= min_true:
            preds.append(p)
            trues.append(t)
    return preds, trues


def get_user_skill_dict(skill_sets, user_sets):
    user_skill = {}
    for skills, users in zip(skill_sets, user_sets):
        for u in users:
            if u not in user_skill.keys():
                user_skill[u] = []
            user_skill[u].extend(skills)
    return user_skill


def get_skill_sets():
    with open('./baselineOutputs/baseline_skill_test.csv') as f:
        skill_sets = []
        for line in f.readlines():
            skill_sets.append([s.strip() for s in line.split(',')])
    return skill_sets


authorNameIds = pandas.read_csv('./baselineOutputs/authorNameId_{}.txt'.format(year), encoding='utf_8', header=None, delimiter='	', names=["NameID", "Author"])
with open('./baselineOutputs/test_authors_{}.csv'.format(year), 'r') as f:
    predictions = []
    lines = f.readlines()
    for line in lines:
        authors = [authorNameIds.loc[authorNameIds['NameID']==int(author)]['Author'].values[0].strip().lower() for author in line.split(',')[1:]]
        if authors.__len__() < k:
            diff = k - authors.__len__()
            authors += ['-1'] * diff
        predictions.append(authors)
predictions = np.asarray(predictions)

y_test = []
dataset = load_preprocessed_dataset()
authorNames_true, _ = load_authors('../dataset/authorNameId.txt')
authorNames_true = [author.strip().lower() for author in authorNames_true]
authorNames_true = np.asarray(authorNames_true)
testIndices = load_train_test_indices()[1]['Test']
for testIndex in testIndices:
    for sample in dataset:
        if sample[0] == testIndex:
            authorIDs = sample[2].nonzero()[1]
            y_test.append(authorNames_true[authorIDs])
            continue
y_test = np.asarray(y_test)

k_set = np.arange(1, k+1, 1)
r_at_k = dblp_eval.init_eval_holder(k_set) # all r@k of instances in one fold and one k_evaluation_set
r_at_k_all = dblp_eval.init_eval_holder(k_set) # all r@k of instances in one fold and one k_evaluation_set
mapk = dblp_eval.init_eval_holder(k_set) # all r@k of instances in one fold and one k_evaluation_set
ndcg = dblp_eval.init_eval_holder(k_set) # all r@k of instances in one fold and one k_evaluation_set
mrr = dblp_eval.init_eval_holder(k_set) # all r@k of instances in one fold and one k_evaluation_set
tf_score = dblp_eval.init_eval_holder(k_set)  # all r@k of instances in one fold and one k_evaluation_set
pred_idx, test_idx = filter_len(predictions, y_test)
pred_idx_list = [list(y) for y in pred_idx]
test_idx_list = [list(y) for y in test_idx]
pred_idx = np.asarray(pred_idx)
test_idx = np.asarray(test_idx)
user_skill_dict = get_user_skill_dict(get_skill_sets(), y_test)
for target_k in k_set:
    all_recall_mean, all_recall = calc_r_at_k(pred_idx[:, :target_k], test_idx)
    r_at_k[target_k].append(all_recall_mean)
    r_at_k_all[target_k].append(all_recall)
    mapk[target_k].append(metrics.mapk(test_idx_list, pred_idx_list, k=target_k))
    ndcg[target_k].append(rk.ndcg_at(pred_idx_list, test_idx_list, k=target_k))
    mrr[target_k].append(dblp_eval.mean_reciprocal_rank(dblp_eval.cal_relevance_score(pred_idx_list[:][:target_k], test_idx_list)))
    tf_score[target_k].append(dblp_eval.team_formation_feasiblity(predictions, y_test, user_skill_dict, target_k))

plotter.plot_at_k(k_set, r_at_k, 'Recall@k')



with open('./Baseline_{}_r_at_k_50.pkl'.format(year), 'wb') as f:
    pkl.dump(r_at_k, f)

with open('./Baseline_{}_r_at_k_all_50.pkl'.format(year), 'wb') as f:
    pkl.dump(r_at_k_all, f)

with open('./Baseline_{}_mapk_50.pkl'.format(year), 'wb') as f:
    pkl.dump(mapk, f)

with open('./Baseline_{}_ndcg_50.pkl'.format(year), 'wb') as f:
    pkl.dump(ndcg, f)

with open('./Baseline_{}_mrr_50.pkl'.format(year), 'wb') as f:
    pkl.dump(mrr, f)

with open('./Baseline_{}_tf_50.pkl'.format(year), 'wb') as f:
    pkl.dump(tf_score, f)

