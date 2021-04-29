from os import path
import pandas
import pickle
from teamFormationLibrary.data_access_layer import *
import teamFormationLibrary.eval.evaluator as dblp_eval


def get_user_skill_dict(data):
    dict = {}
    for sample in data:
        id = sample[0]
        skill = sample[1].nonzero()[1]
        user = sample[2].nonzero()[1]
        for u in user:
            if u not in dict.keys():
                dict[u] = []
            dict[u].extend(skill)
    return dict


def get_foldIDsampleID_stata_dict(data, train_test_indices, kfold=10):
    evaluation_k_set = np.arange(1, kfold + 1, 1)
    foldIDsampleID_stata_dict = dblp_eval.init_eval_holder(evaluation_k_set)
    for fold_counter in evaluation_k_set:
        _, _, x_test, _ = get_fold_data(fold_counter, data, train_test_indices, mute=True)
        for smaple in x_test:
            foldIDsampleID_stata_dict[fold_counter].append(len(smaple[0].nonzero()[1]))
    return foldIDsampleID_stata_dict


def nn_t2v_dataset_generator(model, dataset, output_file_path, mode='user'):
    t2v_dataset = []
    counter = 1
    for record in dataset:
        id = record[0]
        if mode.lower() == 'user':
            try:
                skill_vec = record[1].todense()
                team_vec = model.get_team_vec(id)
                t2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'skill':
            try:
                skill_vec = model.get_team_vec(id)
                team_vec = record[2].todense()
                t2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'full':
            try:
                model_skill = model['skill']
                model_user = model['user']
                skill_vec = model_skill.get_team_vec(id)
                team_vec = model_user.get_team_vec(id)
                t2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
    with open(output_file_path, 'wb') as f:
        pickle.dump(t2v_dataset, f)


def get_fold_data(fold_counter, dataset, train_test_indices, mute=False):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_index = train_test_indices[fold_counter]['Train']
    test_index = train_test_indices[fold_counter]['Test']
    for sample in dataset:
        id = sample[0]
        if id in train_index:
            x_train.append(sample[1])
            y_train.append(sample[2])
        elif id in test_index:
            x_test.append(sample[1])
            y_test.append(sample[2])

    x_train = np.asarray(x_train).reshape(len(x_train), -1)
    y_train = np.asarray(y_train).reshape(len(y_train), -1)
    x_test = np.asarray(x_test).reshape(len(x_test), -1)
    y_test = np.asarray(y_test).reshape(len(y_test), -1)

    if not mute:
        print('Fold number {}'.format(fold_counter))
        print('dataset Size: {}'.format(len(dataset)))
        print('Train Size: {} Test Size: {}'.format(x_train.__len__(), x_test.__len__()))
    return x_train, y_train, x_test, y_test


def load_train_test_indices(file_path='dataset/Train_Test_indices.pkl'):
    with open(file_path, 'rb') as f:
        indices = pickle.load(f)
    return indices


def preprocessed_dataset_exist(file_path='dataset/dblp_preprocessed_dataset.pkl'):
    if path.exists(file_path):
        return True
    return False


def load_preprocessed_dataset(file_path='dataset/dblp_preprocessed_dataset.pkl'): #'dataset/dblp_preprocessed_dataset.pkl'
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def get_user_HIndex(file_path='dataset/authorHIndex.txt'):
    user_hindex_dict = {}
    user_hindex = pandas.read_csv(file_path, encoding='utf_8', header=None, delimiter='	')
    user_hindex = (user_hindex.iloc[:, :2]).values
    for item in user_hindex:
        user_hindex_dict[item[0]] = item[1]
    return user_hindex_dict
