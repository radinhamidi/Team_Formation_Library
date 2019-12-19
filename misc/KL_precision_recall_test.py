from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import eval.evaluator as dblp_eval
import eval. plotter
from cmn.utils import crossValidate
from eval import plotter

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

# Get the train and test as done in DL assignments.
# Train/Validation/Test version
x_train, x_validate, x_test, ids = crossValidate(x, train_ratio, validation_ratio)
y_train = y[ids[0:int(y.__len__() * train_ratio)]]
y_validate = y[ids[int(y.__len__() * train_ratio):int(y.__len__() * (train_ratio + validation_ratio))]]
y_test = y[ids[int(y.__len__() * (train_ratio + validation_ratio)):]]




input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
print("Input/output dimensions ", input_dim, output_dim)
# this is our input placeholder
input_img = Input(shape=(input_dim,))
lambda_val = 0.001  # Weight decay , refer : https://stackoverflow.com/questions/44495698/keras-difference-between-kernel-and-activity-regularizers

# Custom Regularizer function
def sparse_reg(activ_matrix):
    p = 0.01
    beta = 3
    p_hat = K.mean(activ_matrix)  # average over the batch samples
    print("p_hat = ", p_hat)
    # KLD = p*(K.log(p)-K.log(p_hat)) + (1-p)*(K.log(1-p)-K.log(1-p_hat))
    KLD = p * (K.log(p / p_hat)) + (1 - p) * (K.log((1 - p) / (1 - p_hat)))
    print("KLD = ", KLD)
    return beta * K.sum(KLD)  # sum over the layer units


encoded = Dense(encoding_dim,
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(lambda_val / 2), activity_regularizer=sparse_reg)(input_img)

decoded = Dense(output_dim,
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(lambda_val / 2), activity_regularizer=sparse_reg)(
    encoded)  # Switch to softmax here?

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adagrad', loss='mean_absolute_error')  # What optimizer to use??


# Removing normalization since using MSE - later
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.load_weights('./KL.h5')

# Unit test for recall and precision @k methods

y_pred = autoencoder.predict(x_test)

k_set = np.arange(1, 51, 1)
p_at_k = dblp_eval.init_eval_holder(k_set) # all p@k of instances in one fold and one k_evaluation_set
r_at_k = dblp_eval.init_eval_holder(k_set) # all p@k of instances in one fold and one k_evaluation_set
for k in k_set:
    print('Calculating for @{}'.format(k))
    # all_precision = []
    # for pred, t in zip(y_pred, y_test):
    #     t = np.asarray(t)
    #     pred = np.asarray(pred)
    #
    #     t_indices = np.argwhere(t)
    #     if t_indices.__len__() == 0:
    #         continue
    #     pred_indices = pred.argsort()[-k:][::-1]
    #
    #     precision = 0
    #     for pred_index in pred_indices:
    #         if pred_index in t_indices:
    #             precision += 1
    #     all_precision.append(precision/pred_indices.__len__())

    all_recall = []
    for pred, t in zip(y_pred, y_test):
        t = np.asarray(t)
        pred = np.asarray(pred)

        t_indices = np.argwhere(t)
        if t_indices.__len__() == 0:
            continue
        pred_indices = pred.argsort()[-k:][::-1]

        recall = 0
        for t_index in t_indices:
            if t_index in pred_indices:
                recall += 1
        all_recall.append(recall/t_indices.__len__())

    # p_at_k[k] = np.asarray(all_precision).mean()
    r_at_k[k] = np.asarray(all_recall).mean()


plotter.plot_at_k(k_set, r_at_k, 'Recall@k')
# plotter.plot_at_k(k_set, p_at_k, 'Precision@k')


r_at_k_name = str(input('File name for saving R@K?'))
with open('./{}.pkl'.format(r_at_k_name), 'wb') as f:
    pkl.dump(r_at_k, f)