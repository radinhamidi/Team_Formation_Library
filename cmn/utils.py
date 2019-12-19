import json

import numpy as np
import pandas as pd
import pickle
import scipy
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import preprocessing
import itertools
import keras
import glob
from ml.team2vec import *


def crossValidate(data: np.ndarray, split1, split2):
    data = np.asarray(data)
    m = data.__len__()
    idx = np.random.permutation(m)
    data = data[idx]
    return data[:int(split1 * m), :], data[int(split1 * m):int((split1 + split2) * m), :], data[
                                                                                           int((split1 + split2) * m):,
                                                                                           :], idx


def kfold(data: np.ndarray, k):
    m = data.__len__()
    idx = np.random.permutation(m)
    kfoldData = []
    for i in range(k):
        training = [x for j, x in enumerate(idx) if j % k != i]
        validation = [x for j, x in enumerate(idx) if j % k == i]
        train = data[training]
        test = data[validation]
        kfoldData.append([train, test])
    return np.asarray(kfoldData), idx


def scale(x):
    return preprocessing.scale(x)


def SVD_compress(x, accuracy):
    m, n = x.shape
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
    print("Dimension reduction with SVD went from {} to {} when {}% of eigen values have already saved.".format(n,
                                                                                                                featurelength,
                                                                                                                accuracy * 100))
    return U * S, [U, S, V]


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
    # alternate cmap : YlOrRd
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


def pick_model_weights(model: keras.models.Model, dataset_name, path='../output/Models/Weights'):
    model_names = []
    for path in glob.glob('{}/{}_*.h5'.format(path, dataset_name)):
        model_names.append(path)
    print('Please enter you model number form list below:')
    for i, path in enumerate(model_names):
        print('{}. {}'.format(i + 1, path))
    model_number = int(input('?')) - 1
    model.load_weights(model_names[model_number])
    return model


def load_T2V_model(model: Team2Vec, path='../output/Models/T2V'):
    model_names = []
    for path in glob.glob('{}/model_*'.format(path)):
        model_names.append(path)
    print('Please enter you model number form list below:')
    for i, path in enumerate(model_names):
        print('{}. {}'.format(i + 1, path))
    model_number = int(input('?')) - 1
    model.load_model(model_names[model_number])
    return model


def save_record(dict, dict_name, dir='../output/'):
    json.dump(str(dict), open(dir + dict_name + '.json', 'w'))