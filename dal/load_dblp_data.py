from os import path
from scipy import sparse
import pandas
from collections import Counter
import numpy as np
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from ml.team2vec import *

publication_filter = ['sigmod', 'vldb', 'icde', 'icdt', 'edbt', 'pods', 'kdd', 'www',
                      'sdm', 'pkdd', 'icdm', 'cikm', 'aaai', 'icml', 'ecml', 'colt',
                      'uai', 'soda', 'focs', 'stoc', 'stacs']
'''
currentArticle.venue.toLowerCase().contains("sigmod") ||
                                currentArticle.venue.toLowerCase().contains("vldb") ||
                                currentArticle.venue.toLowerCase().contains("icde") ||
                                currentArticle.venue.toLowerCase().contains("icdt") ||
                                currentArticle.venue.toLowerCase().contains("edbt") ||
                                currentArticle.venue.toLowerCase().contains("pods") ||
                                currentArticle.venue.toLowerCase().contains("kdd") ||
                                currentArticle.venue.toLowerCase().contains("www") ||
                                currentArticle.venue.toLowerCase().contains("sdm") ||
                                currentArticle.venue.toLowerCase().contains("pkdd") ||
                                currentArticle.venue.toLowerCase().contains("icdm") ||
                                currentArticle.venue.toLowerCase().contains("cikm") ||
                                currentArticle.venue.toLowerCase().contains("aaai") ||
                                currentArticle.venue.toLowerCase().contains("icml") ||
                                currentArticle.venue.toLowerCase().contains("ecml") ||
                                currentArticle.venue.toLowerCase().contains("colt") ||
                                currentArticle.venue.toLowerCase().contains("uai") ||
                                currentArticle.venue.toLowerCase().contains("soda") ||
                                currentArticle.venue.toLowerCase().contains("focs") ||
                                currentArticle.venue.toLowerCase().contains("stoc") ||
                                currentArticle.venue.toLowerCase().contains("stacs")
//                                currentArticle.venue.toLowerCase().contains("data") ||
//                                currentArticle.venue.toLowerCase().contains("knowledge")
//                                currentArticle.venue.toLowerCase().contains("min") ||
//                                currentArticle.venue.toLowerCase().contains("computer") ||
//                                currentArticle.venue.toLowerCase().contains("system") ||
//                                currentArticle.venue.toLowerCase().contains("anal") ||
//                                currentArticle.venue.toLowerCase().contains("network") ||
//                                currentArticle.venue.toLowerCase().contains("proc") ||
//                                currentArticle.venue.toLowerCase().contains("design") ||
//                                currentArticle.venue.toLowerCase().contains("inter") ||
//                                currentArticle.venue.toLowerCase().contains("exp") ||
//                                currentArticle.venue.toLowerCase().contains("theory") ||
'''


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


def convert_to_pkl(txt_dir='../dataset/dblp.txt', pkl_dir='../dataset/dblp.pkl', ftype='dict'):
    load_dblp_arnet(txt_dir, pkl_dir, ftype=ftype)


# dblp to sparse matrix: output: pickle file of the sparse matrix
def extract_data(filter_journals=False, size_limit=np.inf, skill_size_filter=0, member_size_filter=0,
                 source_dir='../dataset/dblp.pkl', skill_dir='../dataset/invertedTermCount.txt',
                 author_dir='../dataset/authorNameId.txt', output_dir='../dataset/ae_dataset.pkl'):
    if not source_pkl_exist(file_path=source_dir):
        convert_to_pkl()

    data = load_citation_pkl(source_dir)
    skills, skills_freq = load_skills(skill_dir)
    authors, nameIDs = load_authors(author_dir)
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


def load_ae_dataset(file_path='../dataset/ae_dataset.pkl'):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


# we should check if we have the proper dataset for ae or not. and if not then we run extract_data function
def ae_data_exist(file_path='../dataset/ae_dataset.pkl'):
    if path.exists(file_path):
        return True
    else:
        return False


# we should check if we have the source dataset or not. and if not then we run convert_to_pkl() function
def source_pkl_exist(file_path='../dataset/dblp.pkl'):
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

def nn_t2v_dataset_generator(model: Team2Vec, dataset, output_file_path):
    t2v_dataset = []
    counter = 1
    for record in dataset:
        id = record[0]
        try:
            skill_vec = record[1].todense()
            team_vec = model.get_team_vec(id)
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

def dataset_preprocessing(dataset, min_records=10, kfolds=10, max_features=2000, n_gram=3,
                          dataset_source_dir='../dataset/dblp.pkl', shuffle_at_the_end=False,
                          save_to_pkl=True, indices_dict_file_path='../dataset/Train_Test_indices.pkl',
                          preprocessed_dataset_file_path='../dataset/dblp_preprocessed_dataset.pkl', seed=7):
    random.seed(seed)
    author_paper_counter = Counter()
    author_docID_dict = {}
    docID_author_dict = {}

    for sample in dataset:
        id = sample[0]
        sparse_authorIDs = sparse.coo_matrix(sample[2])
        author_ids = sparse_authorIDs.nonzero()[1]
        for author_id in author_ids:
            if author_id not in author_docID_dict.keys():
                author_docID_dict[author_id] = []
            author_docID_dict[author_id].append(id)
        author_paper_counter.update(author_ids)

    eligible_authors = [x for x in author_paper_counter.keys() if author_paper_counter.get(x) >= min_records]

    eligible_documents = []
    for eligible_author in eligible_authors:
        eligible_documents.extend(author_docID_dict[eligible_author])
    eligible_documents = np.unique(eligible_documents).tolist()

    data = load_citation_pkl(dataset_source_dir)
    data = np.asarray(data)
    eligible_titles = []
    for eligible_paper in data[eligible_documents]:
        eligible_titles.append(eligible_paper['title'].strip())
    vect = TfidfVectorizer(tokenizer=tokenize, analyzer='word', lowercase=True, stop_words='english',
                           ngram_range=(1, n_gram), max_features=max_features)
    vect.fit(eligible_titles)

    for sample in dataset:
        id = sample[0]
        if id in eligible_documents:
            sparse_authorIDs = sparse.coo_matrix(sample[2])
            author_ids = sparse_authorIDs.nonzero()[1]
            docID_author_dict[id] = author_ids

    author_set = []
    skill_sets = []
    id_sets = []
    for eligible_document_id in eligible_documents:
        skill_set = vect.transform([data[eligible_document_id]['title'].strip()])
        skill_set.tocoo()
        if skill_set.count_nonzero() > 0:
            author_set.extend(docID_author_dict[eligible_document_id])
            id_sets.append(eligible_document_id)
            skill_sets.append(skill_set)
        else:
            eligible_documents.remove(eligible_document_id)
            del docID_author_dict[eligible_document_id]
    author_set = np.unique(author_set).tolist()

    print('Number of eligible documents equal/more than {} records: {}'.format(min_records, len(eligible_documents)))
    print('Number of eligible authors equal/more than {} records: {}'.format(min_records, len(eligible_authors)))

    preprocessed_dataset = []
    for id, skill_vector in zip(id_sets, skill_sets):
        author_vector = np.zeros(len(eligible_authors))
        for author_id in docID_author_dict[id]:
            if author_id in eligible_authors:
                author_vector[eligible_authors.index(author_id)] = 1
        author_vector = sparse.coo_matrix(author_vector)
        preprocessed_dataset.append([id, skill_vector, author_vector])

    if save_to_pkl:
        with open('{}'.format(preprocessed_dataset_file_path), 'wb') as f:
            pkl.dump(preprocessed_dataset, f)

    indices = split_data(kfolds, author_docID_dict, eligible_documents, shuffle_at_the_end, save_to_pkl, indices_dict_file_path)

    return indices, preprocessed_dataset

def split_data(kfolds, author_docID_dict, eligible_documents, shuffle_at_the_end, save_to_pkl, indices_dict_file_path):
    train_docs = []
    test_docs = []
    rule_violence_counter = 0
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

    print("Number of Train docs: {}".format(len(train_docs)))
    print("Number of Test docs: {}".format(len(test_docs)))
    print("Number of violations in train/test split because of already"
          " existence of a paper from target author in test set: {}".format(rule_violence_counter))

    if shuffle_at_the_end:
        train_docs = random.shuffle(train_docs)
        test_docs = random.shuffle(test_docs)

    indices = {1: {'Train': train_docs, 'Test': test_docs}}

    if save_to_pkl:
        with open('{}'.format(indices_dict_file_path), 'wb') as f:
            pkl.dump(indices, f)

    return indices

def get_fold_data(fold_counter, dataset, train_test_indices):
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

    print('Fold number {}'.format(fold_counter))
    print('dataset Size: {}'.format(len(dataset)))
    print('Train Size: {} Test Size: {}'.format(x_train.__len__(), x_test.__len__()))
    return x_train, y_train, x_test, y_test

def train_test_indices_exist(file_path='../dataset/Train_Test_indices.pkl'):
    if path.exists(file_path):
        return True
    return False

def load_train_test_indices(file_path='../dataset/Train_Test_indices.pkl'):
    with open(file_path, 'rb') as f:
        indices = pickle.load(f)
    return indices

def preprocessed_dataset_exist(file_path='../dataset/dblp_preprocessed_dataset.pkl'):
    if path.exists(file_path):
        return True
    return False

def load_preprocessed_dataset(file_path='../dataset/dblp_preprocessed_dataset.pkl'):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset