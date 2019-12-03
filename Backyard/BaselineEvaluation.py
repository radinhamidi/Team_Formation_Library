import pickle as pkl
from DataAccessLayer.load_dblp_data import *
from Common.Utils import crossValidate
import Evaluation.Evaluator as dblp_eval
from Evaluation import plotter

seed = 7
np.random.seed(seed)


fax = './x_sampleset.pkl'
fay = './y_sampleset.pkl'
with open(fax, 'rb') as f:
    x = pkl.load(f)
with open(fay, 'rb') as f:
    y = pkl.load(f)

# Variables
train_ratio = 0.8
validation_ratio = 0.0
epochs = 30
batch_sizes = 8
sp = 0.01
b_val = 3  # Controls the acitvity of the hidden layer nodes
encoding_dim = 3000

### Get the train and test as done in DL assignments.
# Train/Validation/Test version
x_train, x_validate, x_test, ids = crossValidate(x, train_ratio, validation_ratio)
y_train = y[ids[0:int(y.__len__() * train_ratio)]]
y_validate = y[ids[int(y.__len__() * train_ratio):int(y.__len__() * (train_ratio + validation_ratio))]]
y_test = y[ids[int(y.__len__() * (train_ratio + validation_ratio)):]]

input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
print("Input/Output dimensions ", input_dim, output_dim)



### Writing target teams skills into file
# skill_dir='../Dataset/invertedTermCount.txt'
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
        t = np.asarray(t)
        pred_indices = np.asarray(pred)

        t_indices = np.argwhere(t)
        if t_indices.__len__() == 0:
            continue

        recall = 0
        for t_index in t_indices:
            if t_index in pred_indices:
                recall += 1
        all_recall.append(recall/t_indices.__len__())
    return np.mean(all_recall), all_recall


k = 50
with open('./test_authors.csv', 'r') as f:
    lis = []
    for line in f:
        authors = [int(author) for author in line.split(',')]
        authors = authors[1:]
        if authors.__len__() < k:
            diff = k - authors.__len__()
            authors += [-1] * diff
        lis.append(authors)

lis = np.asarray(lis)

k_set = np.arange(1, 51, 1)
r_at_k = dblp_eval.init_eval_holder(k_set) # all r@k of instances in one fold and one k_evaluation_set
for target_k in k_set:
    all_recall_mean, all_recall = calc_r_at_k(lis[:, :target_k], y_test)
    r_at_k[target_k] = all_recall_mean

plotter.plot_at_k(k_set, r_at_k, 'Recall@k')


r_at_k_name = str(input('File name for saving R@K?'))
with open('./{}.pkl'.format(r_at_k_name), 'wb') as f:
    pkl.dump(r_at_k, f)