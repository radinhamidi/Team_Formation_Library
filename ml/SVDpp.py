import dal.load_dblp_data as dblp
import surprise as sr
from collections import defaultdict
import numpy as np
import eval.evaluator as dblp_eval
import pickle as pkl

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#running settings
dataset_name = 'DBLP'
method_name = 'SVDpp'

#eval settings
k_fold = 10
k_max = 50 #cut_off for eval
evaluation_k_set = np.arange(1, k_max+1, 1)

#nn settings
epochs = 150
back_propagation_batch_size = 64
training_batch_size = 6000
min_skill_size = 0
min_member_size = 0
latent_dim = 100


if dblp.preprocessed_dataset_exist() and dblp.train_test_indices_exist():
    train_test_indices = dblp.load_train_test_indices()
else:
    if not dblp.ae_data_exist(file_path='../dataset/ae_dataset.pkl'):
        dblp.extract_data(filter_journals=True, skill_size_filter=min_skill_size, member_size_filter=min_member_size)
    if not dblp.preprocessed_dataset_exist() or not dblp.train_test_indices_exist():
        dblp.dataset_preprocessing(dblp.load_ae_dataset(file_path='../dataset/ae_dataset.pkl'), seed=seed, kfolds=k_fold, shuffle_at_the_end=True)
    dblp.load_preprocessed_dataset()
    train_test_indices = dblp.load_train_test_indices()

fold_counter = 1
preprocessed_dataset = dblp.load_preprocessed_dataset()
x_train, y_train, x_test, y_test = dblp.get_fold_data(fold_counter, preprocessed_dataset, train_test_indices)
df_train = dblp.create_user_item(x_train, y_train)
reader = sr.Reader(rating_scale=(1, 1))
data_train = sr.Dataset.load_from_df(df_train[['userID', 'itemID', 'rating']], reader)
df_test = dblp.create_user_item(x_test, y_test)
data_test_temp = sr.Dataset.load_from_df(df_test[['userID', 'itemID', 'rating']], reader)

temp=data_test_temp.build_full_trainset()
data_test = temp.build_anti_testset()

algo = sr.SVDpp()
algo.fit(data_train.build_full_trainset())

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

predictions = algo.test(data_test)
r_at_k = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
p_at_k = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
for k in evaluation_k_set:
    precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=0.5)
    rak = np.mean(list(recalls.values()))
    pak = np.mean(list(precisions.values()))
    r_at_k[k].append(rak)
    p_at_k[k].append(pak)
    print("For top {} in train data: R@{}:{}".format(k, k, rak))
    print("For top {} in train data: P@{}:{}".format(k, k, pak))

compare_submit = input('Submit for compare? (y/n)')
if compare_submit.lower() == 'y':
    with open('../misc/{}_r_at_k_50.pkl'.format(method_name), 'wb') as f:
        pkl.dump(r_at_k, f)
    with open('../misc/{}_p_at_k_50.pkl'.format(method_name), 'wb') as f:
        pkl.dump(p_at_k, f)