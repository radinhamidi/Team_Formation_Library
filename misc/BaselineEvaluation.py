import csv
from dal.load_dblp_data import *
import time

seed = 7
np.random.seed(seed)
k_fold = 10
min_true = 1

year = 2009
# year = 2017
method_name = 'BL{}'.format(year)


##### After running baseline code

# Output formatting
time_str = time.strftime("%Y_%m_%d-%H_%M_%S")

authorNames_true = load_authors('../dataset/authorNameId.txt')
authorNameID_true_dict = {i.strip().lower(): x for i, x in zip(authorNames_true[0], authorNames_true[1])}
dataset = load_preprocessed_dataset()
train_test_indices = load_train_test_indices()

with open("../output/predictions/{}_output.csv".format(method_name), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Method Name', '# Total Folds', '# Fold Number', '# Predictions', '# Truth', 'Computation Time (ms)',
         'Prediction Indices', 'True Indices'])

    for fold_counter in range(1, k_fold + 1):
        _, _, _, y_test = get_fold_data(fold_counter, dataset, train_test_indices)
        true_indices = [i[0].nonzero()[1].tolist() for i in y_test]
        authorNameIds_pred = pandas.read_csv('./baselineOutputs/{}/authorNameId_{}.txt'.format(year, fold_counter),
                                             encoding='utf_8', header=None, delimiter='	',
                                             names=["NameID", "Author"])
        with open('./baselineOutputs/{}/test_authors_{}.csv'.format(year, fold_counter), 'r') as f:
            lines = f.readlines()
            for line, true_index in zip(lines, true_indices):
                splitted = line.split(',')
                method_name = splitted[0]
                # k_fold = int(splitted[1])
                fold_number = int(splitted[2])
                prediction_number = int(splitted[3])
                elapsed_time = float(splitted[4])
                prediction_index = [int(i) for i in splitted[5:]]
                authors = [authorNameIds_pred.loc[authorNameIds_pred['NameID'] == int(author)]['Author'].values[
                               0].strip().lower() for author in prediction_index[5:]]
                pred_index = [int(authorNameID_true_dict[i]) for i in authors]
                if len(pred_index) < prediction_number:
                    diff = prediction_number - len(pred_index)
                    pred_index += ['-1'] * diff
                if len(true_index) >= min_true:
                    writer.writerow([method_name, k_fold, fold_counter, len(pred_index), len(true_index),
                                    elapsed_time] + pred_index + true_index)
                f.close()
    file.close()
