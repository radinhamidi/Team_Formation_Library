import pickle as pkl
from dal.load_dblp_data import *
from cmn.utils import crossValidate
import eval.evaluator as dblp_eval
from eval import plotter
import ml_metrics as metrics

seed = 7
np.random.seed(seed)
k = 50

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


authorNameIds = pandas.read_csv('./baselineOutputs/authorNameId.txt', encoding='utf_8', header=None, delimiter='	', names=["NameID", "Author"])
with open('./baselineOutputs/test_authors.csv', 'r') as f:
    predictions = []
    lines = f.readlines()
    for line in lines:
        authors = [authorNameIds.loc[authorNameIds['NameID']==int(author)]['Author'].values[0].strip().lower() for author in line.split(',')[1:]]
        if authors.__len__() < k:
            diff = k - authors.__len__()
            authors += ['-1'] * diff
        predictions.append(authors)
predictions_list = predictions.copy()
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
y_test_list = [list(y) for y in y_test]
y_test = np.asarray(y_test)

k_set = np.arange(1, k+1, 1)
r_at_k = dblp_eval.init_eval_holder(k_set) # all r@k of instances in one fold and one k_evaluation_set
mapk = dblp_eval.init_eval_holder(k_set) # all r@k of instances in one fold and one k_evaluation_set
for target_k in k_set:
    all_recall_mean, all_recall = calc_r_at_k(predictions[:, :target_k], y_test)
    r_at_k[target_k] = all_recall_mean
    mapk[target_k] = metrics.mapk(y_test_list, predictions_list, k=target_k)

plotter.plot_at_k(k_set, r_at_k, 'Recall@k')

with open('./Baseline_2009_r_at_k_50.pkl', 'wb') as f:
    pkl.dump(r_at_k, f)

with open('./Baseline_2009_mapk_50.pkl', 'wb') as f:
    pkl.dump(mapk, f)