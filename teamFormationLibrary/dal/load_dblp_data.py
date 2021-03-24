import csv
from os import path
from scipy import sparse
import pandas
from collections import Counter
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from teamFormationLibrary.data_access_layer import * #replaced
import matplotlib.pyplot as plt
import xlwt
import teamFormationLibrary.eval.evaluator as dblp_eval
from tempfile import TemporaryFile
from iteration_utilities import deepflatten
from itertools import groupby


publication_filter = ['sigmod', 'vldb', 'icde', 'icdt', 'edbt', 'pods', 'kdd', 'www',
                      'sdm', 'pkdd', 'icdm', 'cikm', 'aaai', 'icml', 'ecml', 'colt',
                      'uai', 'soda', 'focs', 'stoc', 'stacs']


def load_dblp_arnet(infname, outfname, ftype='dict'):  # source: https://gist.github.com/cntswj/51d3379692fd5e553cb6
    # dictionary version added to it
    if ftype == 'dict':
        with open(infname, 'r', encoding='utf-8') as f:
            output = []
            count = 0
            for key, group in groupby(f, key=lambda l: l.strip(' \n\r') == ''):
                if not key:
                    refs = []
                    authors = []
                    title, venue, year, idx, abstract = [''] * 5
                    for item in group:
                        item = item.strip(' \r\n')
                        if item.startswith('#*'):
                            title = item[2:]
                        elif item.startswith('#@'):
                            authors = item[2:].split(',')
                        elif item.startswith('#t'):
                            year = item[2:]
                        elif item.startswith('#c'):
                            venue = item[2:]
                        elif item.startswith('#index'):
                            idx = item[6:]
                        elif item.startswith('#!'):
                            abstract = item[2:]
                        elif item.startswith('#%'):
                            refs.append(item[2:])
                    output.append(
                        {'idx': idx, 'title': title, 'venue': venue, 'authors': authors, 'year': year, 'refs': refs,
                         'abstract': abstract})
                    count += 1
                    print('\r%d\tlines' % (count,), )
        with open(outfname, 'wb') as f:
            pickle.dump(output, f)
    elif ftype == 'csv':
        with open(infname, 'r', encoding='utf-8') as f, open(outfname, 'w', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=',',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            count = 0
            for key, group in groupby(f, key=lambda l: l.strip(' \n\r') == ''):
                if not key:
                    refs = []
                    authors = []
                    title, venue, year, idx, abstract = [''] * 5
                    for item in group:
                        item = item.strip(' \r\n')
                        if item.startswith('#*'):
                            title = item[2:]
                        elif item.startswith('#@'):
                            authors = item[2:].split(',')
                        elif item.startswith('#t'):
                            year = item[2:]
                        elif item.startswith('#c'):
                            venue = item[2:]
                        elif item.startswith('#index'):
                            idx = item[6:]
                        elif item.startswith('#!'):
                            abstract = item[2:]
                        elif item.startswith('#%'):
                            refs.append(item[2:])
                    csv_writer.writerow(
                        [idx, title, venue, authors, year, refs, abstract])
                    count += 1
                    print('\r%d\tlines' % (count,), )


def load_citation_csv(input_file):
    data = np.recfromcsv(input_file, encoding='utf_8')
    return data


def load_citation_pkl(input_file):
    with open(input_file, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data


def load_skills(dir):
    skills_counts = pandas.read_csv(dir, encoding='utf_8', header=None, delimiter='	')
    skills = skills_counts.iloc[:, 0]
    skills_frequency = skills_counts.iloc[:, 1]
    return skills, skills_frequency


def load_authors(dir):
    authorNameIds = pandas.read_csv(dir, encoding='utf_8', header=None, delimiter='	', names=["NameID", "Author"])
    authorNameIds_sorted = authorNameIds.sort_values(by='NameID')
    nameIDs = authorNameIds_sorted.iloc[:, 0]
    authors = authorNameIds_sorted.iloc[:, 1]
    return authors.values, nameIDs.values


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


def convert_to_pkl(txt_dir='dataset/dblp.txt', pkl_dir='dataset/dblp.pkl', ftype='dict'):
    load_dblp_arnet(txt_dir, pkl_dir, ftype=ftype)


# dblp to sparse matrix: output: pickle file of the sparse matrix
def extract_data(filter_journals=True, size_limit=np.inf, skill_size_filter=0, member_size_filter=0,
                 source_dir='dataset/dblp.pkl', skill_dir='dataset/invertedTermCount.txt',
                 author_dir='dataset/authorNameId.txt', output_dir='dataset/ae_dataset.pkl'):
    if not source_pkl_exist(file_path=source_dir):
        convert_to_pkl()

    data = load_citation_pkl(source_dir)
    skills, skills_freq = load_skills(skill_dir)
    authors, _ = load_authors(author_dir)
    skills = np.asarray(skills)
    authors = [author.strip().lower() for author in authors]

    dataset = []
    counter = 0
    id = 0
    for record in data:
        id += 1
        if filter_journals and not filter_pubs(record['venue']):
            continue
        skill_vector = np.zeros(skills.__len__())
        user_vector = np.zeros(authors.__len__())

        for author in record['authors']:
            try:
                user_vector[authors.index(author.strip().lower())] = 1
            except:
                pass

        for i, s in enumerate(skills):
            if s in record['title']:
                skill_vector[i] = 1

        if np.sum(skill_vector) <= skill_size_filter or np.sum(user_vector) <= member_size_filter:
            continue

        skill_vector_sparse = sparse.coo_matrix(skill_vector)
        user_vector_sparse = sparse.coo_matrix(user_vector)
        dataset.append([id, skill_vector_sparse, user_vector_sparse])
        print("File {} added to dataset file. (Total Files: {})".format(id, counter + 1))

        counter += 1
        if counter >= size_limit:
            break

    with open(output_dir, 'wb') as f:
        pickle.dump(dataset, f)
        print('{} records saved to {} successfully.'.format(counter + 1, output_dir))


def load_ae_dataset(file_path='dataset/ae_dataset.pkl'):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


# we should check if we have the proper dataset for ae or not. and if not then we run extract_data function
def ae_data_exist(file_path='dataset/ae_dataset.pkl'):
    if path.exists(file_path):
        return True
    else:
        return False


# we should check if we have the source dataset or not. and if not then we run convert_to_pkl() function
def source_pkl_exist(file_path='dataset/dblp.pkl'):
    if path.exists(file_path):
        return True
    else:
        return False


# Filter wanted publications by their journal
def filter_pubs(venue: str):
    found = False
    for allowed_pub in publication_filter:
        if allowed_pub.lower() in venue.lower():
            found = True
            break
    return found


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


def get_memebrID_by_teamID(preds_ids):
    dataset = load_preprocessed_dataset()
    dataset = pandas.DataFrame(dataset, columns=['id', 'skill', 'author'])
    preds_authors_ids = []
    for pred_ids in preds_ids:
        authors_ids = []
        for id in pred_ids:
            try:
                found_record = dataset.loc[dataset['id'] == id]['author']
                sparse_authors = sparse.coo_matrix(found_record.values.all())
                author_ids = sparse_authors.nonzero()[1]
                authors_ids.extend(author_ids)
            except:
                print('Cannot find team for sample with id: {}'.format(id))
        preds_authors_ids.append(authors_ids)
    return preds_authors_ids


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens



def split_data(kfolds, author_docID_dict, eligible_documents, save_to_pkl, save_to_csv,
               indices_dict_file_path, baseline_path):

    indices = {}
    for fold_counter in range(1,kfolds+1):
        train_docs = []
        test_docs = []
        rule_violence_counter = 0

        for ex_folds in range(1, fold_counter):
            train_docs.extend(indices[ex_folds]['Test'])

        train_docs = list(deepflatten(train_docs))
        c_train = Counter()
        c_train.update(train_docs)
        train_docs = list(c_train.keys())
        train_docs.sort()

        for author in author_docID_dict.keys():
            list_of_author_docs = author_docID_dict.get(author)
            list_of_author_docs = [docID for docID in list_of_author_docs if docID in eligible_documents]

            number_of_test_docs = np.ceil((1 / kfolds) * len(list_of_author_docs))

            already_in_test = []
            for doc in list_of_author_docs:
                if doc in test_docs:
                    already_in_test.append(doc)

            number_of_moving_to_test = int(number_of_test_docs - len(already_in_test))

            if number_of_moving_to_test <= 0:
                rule_violence_counter += 1

            elif number_of_moving_to_test > 0:
                eligible_samples_for_test_set = [doc for doc in list_of_author_docs if
                                                 doc not in already_in_test and doc not in train_docs]
                if len(eligible_samples_for_test_set) > 0:
                    if len(eligible_samples_for_test_set) > number_of_moving_to_test:
                        test_docs.extend(random.sample(eligible_samples_for_test_set, k=number_of_moving_to_test))
                    else:
                        test_docs.extend(eligible_samples_for_test_set)

            train_docs.extend([ele for ele in list_of_author_docs if ele not in test_docs and ele not in train_docs])

        print("******* Fold {} *******".format(fold_counter))
        print("Number of Train docs: {}".format(len(train_docs)))
        print("Number of Test docs: {}".format(len(test_docs)))
        print("Number of violations in train/test split because of already"
              " existence of a paper from target author in test set: {}".format(rule_violence_counter))

        test_docs = list(deepflatten(test_docs))
        c_test = Counter()
        c_test.update(test_docs)
        test_docs = list(c_test.keys())
        test_docs.sort()

        indices[fold_counter] = {'Train': train_docs, 'Test': test_docs}

        if save_to_csv:
            with open('{}baseline_train_indices_{}.csv'.format(baseline_path, fold_counter), 'w') as f:
                f.write(','.join(str(x) for x in train_docs) + '\n')
                f.close()
            with open('{}baseline_test_indices_{}.csv'.format(baseline_path, fold_counter), 'w') as f:
                f.write(','.join(str(x) for x in test_docs) + '\n')
                f.close()

    if save_to_pkl:
        with open('{}'.format(indices_dict_file_path), 'wb') as f:
            pkl.dump(indices, f)

    return indices


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


def train_test_indices_exist(file_path='dataset/Train_Test_indices.pkl'):
    if path.exists(file_path):
        return True
    return False


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


def create_user_item(x, y):
    itemID = []
    userID = []
    rating = []
    for skills, users in zip(x, y):
        skills_ids = list(skills.all().nonzero()[1])
        users_ids = list(users.all().nonzero()[1])
        for skill in skills_ids:
            for user in users_ids:
                itemID.append(skill)
                userID.append(user)
                rating.append(1)
    df = pd.DataFrame({'itemID': itemID, 'userID': userID, 'rating': rating})
    return df


def get_user_HIndex(file_path='dataset/authorHIndex.txt'):
    user_hindex_dict = {}
    user_hindex = pandas.read_csv(file_path, encoding='utf_8', header=None, delimiter='	')
    user_hindex = (user_hindex.iloc[:, :2]).values
    for item in user_hindex:
        user_hindex_dict[item[0]] = item[1]
    return user_hindex_dict


def dataset_histo(min_count=3):
    data = load_preprocessed_dataset()
    user_count = []
    skill_count = []
    for sample in data:
        if len(sample[2].nonzero()[1]) > min_count:
            skill_count.append(len(sample[1].nonzero()[1]))
            user_count.append(len(sample[2].nonzero()[1]))

    plt.figure(0)
    plt.hist(skill_count, bins='auto')
    skill_hist, skill_bins = np.histogram(skill_count, bins=range(30))

    plt.figure(1)
    plt.hist(user_count, bins='auto')
    user_hist, user_bins = np.histogram(user_count, bins=range(20))

    book = xlwt.Workbook()
    sheet1 = book.add_sheet('skill')
    sheet2 = book.add_sheet('user')

    sheet1.write(0,0,'bins')
    sheet1.write(0,1,'quantity')
    for i, e in enumerate(skill_hist):
        sheet1.write(i+1, 0, int(skill_bins[i]))
        sheet1.write(i+1, 1, int(skill_hist[i]))

    sheet2.write(0,0,'bins')
    sheet2.write(0,1,'quantity')
    for i, e in enumerate(user_hist):
        sheet2.write(i+1, 0, int(user_bins[i]))
        sheet2.write(i+1, 1, int(user_hist[i]))

    name = "histogram.xls"
    book.save(name)
    book.save(TemporaryFile())

    plt.show()

    return np.mean(skill_count), np.mean(user_count)

# dataset_histo(min_count=0)


def get_co_occurrence(save_to_file=True):
    data = load_preprocessed_dataset('./dataset/dblp_preprocessed_dataset.pkl')
    mat = np.zeros((data[0][2].shape[1], data[0][2].shape[1]))
    for sample in data:
        indices = sample[2].nonzero()[1]
        for i in indices:
            for j in indices:
                if i != j:
                    mat[i, j] += 1

    if save_to_file:
        np.savetxt('./output/eval_results/co_occurrence.csv', mat, delimiter=',', fmt='%d')

    return mat
