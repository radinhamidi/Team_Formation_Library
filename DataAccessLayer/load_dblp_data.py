from Common.util import *

publication_filter = ['sigmod','vldb','icde','icdt','edbt','pods','kdd','www',
                      'sdm', 'pkdd','icdm','cikm','aaai','icml','ecml','colt',
                      'uai', 'soda','focs','stoc','stacs']
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

def convert_to_pkl(txt_dir = './Dataset/dblp.txt', pkl_dir='./Dataset/dblp.pkl', ftype='dict'):
    load_dblp_arnet(txt_dir, pkl_dir, ftype=ftype)

# dblp to sparse matrix: output: pickle file of the sparse matrix
def extract_data(filter_journals = False, size_limit = np.inf, pkl_dir='./Dataset/dblp.pkl', skill_dir= './Dataset/invertedTermCount.txt',
                 author_dir= './Dataset/authorNameId.txt', output_dir='./Dataset/dataset.pkl'):
    data = load_citation_pkl(pkl_dir)

    skills, skills_freq = load_skills(skill_dir)
    authors, nameIDs = load_authors(author_dir)

    skills = np.asarray(skills)

    dataset = []
    counter = 0
    for record in data:
        if filter_journals:
            if any(keyword in record['venue'] for keyword in publication_filter):
                skill_vector = np.zeros(skills.__len__())
                user_vector = np.zeros(authors.__len__())

                for author in record['authors']:
                    user_vector[np.where(authors == author)] = 1

                for i, s in enumerate(skills):
                    if s in record['title']:
                        skill_vector[i] = 1

                dataset.append([skill_vector, user_vector])
                counter += 1
        else:
            skill_vector = np.zeros(skills.__len__())
            user_vector = np.zeros(authors.__len__())

            for author in record['authors']:
                user_vector[np.where(authors == author)] = 1

            for i, s in enumerate(skills):
                if s in record['title']:
                    skill_vector[i] = 1

            dataset.append([skill_vector, user_vector]) ### ToDo: add instance id
            counter += 1
        if counter >= size_limit:
            break

    with open(output_dir, 'wb') as f:
        pickle.dump(dataset, f)

def load_ae_dataset(dir='./Dataset/dataset.pkl'):
    with open(dir, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def ae_data_exist(): # we should check if we have the proper dataset for ae or not. and if not then we run extract_data function
    return True