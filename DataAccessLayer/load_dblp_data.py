from Common.Utils import *
from os import path
from scipy import sparse

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


def convert_to_pkl(txt_dir='../Dataset/dblp.txt', pkl_dir='../Dataset/dblp.pkl', ftype='dict'):
    load_dblp_arnet(txt_dir, pkl_dir, ftype=ftype)


# dblp to sparse matrix: output: pickle file of the sparse matrix
def extract_data(filter_journals=False, size_limit=np.inf, source_dir='../Dataset/dblp.pkl',
                 skill_dir='../Dataset/invertedTermCount.txt',
                 author_dir='../Dataset/authorNameId.txt', output_dir='../Dataset/ae_dataset.pkl'):

    data = load_citation_pkl(source_dir)
    skills, skills_freq = load_skills(skill_dir)
    authors, nameIDs = load_authors(author_dir)
    skills = np.asarray(skills)

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
            user_vector[np.where(authors == author)] = 1

        for i, s in enumerate(skills):
            if s in record['title']:
                skill_vector[i] = 1

        skill_vector_sparse = sparse.coo_matrix(skill_vector)
        user_vector_sparse = sparse.coo_matrix(user_vector)
        dataset.append([id, skill_vector_sparse, user_vector_sparse])
        print("File {} added to dataset file. (Total Files: {})".format(id, counter+1))

        counter += 1
        if counter >= size_limit:
            break

    with open(output_dir, 'wb') as f:
        pickle.dump(dataset, f)
        print('{} records saved to {} successfully.'.format(counter+1, output_dir))


def load_ae_dataset(dir='../Dataset/ae_dataset.pkl'):
    with open(dir, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


# we should check if we have the proper dataset for ae or not. and if not then we run extract_data function
def ae_data_exist(file_path='../Dataset/ae_dataset.pkl'):
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


def batch_generator(iterable, n=10):
    l = len(iterable)
    for ndx in range(0, l, n):
        batch_length = min(ndx + n, l) - ndx
        yield np.asarray([record.todense() for record in iterable[ndx:min(ndx + n, l)]]).reshape(batch_length, -1)