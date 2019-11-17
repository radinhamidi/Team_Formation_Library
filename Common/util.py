import numpy as np
import pandas as pd
import pickle
import scipy
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import preprocessing
import itertools
import timeit
from enum import Enum,auto
from scipy.special import expit
import csv
from itertools import groupby
import pandas


def crossValidate(data:np.ndarray, split1, split2):
    data = np.asarray(data)
    m = data.__len__()
    idx = np.random.permutation(m)
    data = data[idx]
    return data[:int(split1*m),:], data[int(split1*m):int((split1+split2)*m),:], data[int((split1+split2)*m):,:], idx

def kfold(data:np.ndarray, k):
    m = data.__len__()
    idx = np.random.permutation(m)
    kfoldData = []
    for i in range(k):
        training = [x for j, x in enumerate(idx) if j % k != i]
        validation = [x for j, x in enumerate(idx) if j % k == i]
        train = data[training]
        test = data[validation]
        kfoldData.append([train,test])
    return np.asarray(kfoldData), idx

def scale(x):
    return preprocessing.scale(x)

def SVD_compress(x,accuracy):
    m,n = x.shape
    [U, S, V] = scipy.linalg.svd(x, full_matrices=False)
    # print("Result of close test by numpy package for new data vectors and original one: ",np.allclose(x, np.dot(U * S, V)))
    sigmaSumThreshold = float(accuracy * np.sum(S))
    sum = 0
    i = 0
    while (sum < sigmaSumThreshold):
        sum += S[i]
        i += 1
    featurelength = i
    S = S[0:i]
    U = U[:, 0:i]
    V = V[0:i, :]
    print("Dimension reduction with SVD went from {} to {} when {}% of eigen values have already saved.".format(n,featurelength,accuracy * 100))
    return  U * S, [U,S,V]

def normalize(x):
    x_scaled = deepcopy(x)
    for i in range(x.shape[1]):
        feature = x_scaled[:, i]
        max = np.max(feature)
        min = np.min(feature)
        try:
            x_scaled[:, i] = (feature - min) / (max - min)
        except ZeroDivisionError:
            pass
    return x_scaled

def plot_confusion_matrix(cm, class_names,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #alternate cmap : YlOrRd
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def load_dblp_arnet(infname, outfname, ftype='dict'): # source: https://gist.github.com/cntswj/51d3379692fd5e553cb6
    # dictionary version added to it
    if ftype=='dict':
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
                    output.append({'idx':idx, 'title':title, 'venue':venue, 'authors':authors, 'year':year, 'refs':refs, 'abstract':abstract})
                    count += 1
                    print('\r%d\tlines' % (count,),)
        with open(outfname, 'wb') as f:
            pickle.dump(output, f)
    elif ftype=='csv':
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
                    print('\r%d\tlines' % (count,),)



def load_citation_csv(input_file):
    data = np.recfromcsv(input_file, encoding='utf_8')
    return data


def load_citation_pkl(input_file):
    with open(input_file, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data

def load_skills(dir):
    skills_counts = pandas.read_csv(dir, encoding='utf_8', header=None, delimiter='	')
    skills = skills_counts.iloc[:,0]
    skills_frequency = skills_counts.iloc[:,1]
    return skills, skills_frequency

def load_authors(dir):
    authorNameIds = pandas.read_csv(dir, encoding='utf_8', header=None, delimiter='	', names=["NameID", "Author"])
    authorNameIds_sorted = authorNameIds.sort_values(by='NameID')
    nameIDs = authorNameIds_sorted.iloc[:,0]
    authors = authorNameIds_sorted.iloc[:,1]
    return authors.values, nameIDs.values