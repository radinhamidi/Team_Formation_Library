'''
These are our options:
k-sparse - Done - Did not work on test data - probably not extract concept
L1 regularization - Done - Didnt work that much
KL Regularization - Done - Worked
Custom optimization function
Check CBOW
Other todos:
test precision and recall: Done - they are correct but there is a theoretical bug in p@k since true lables might be
less than top k findings, therefore fome of predictions candidates will be wrong and cause error. Maybe we should set
a upper bound for k equal to number of true candidates
'''

from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import eval.evaluator as dblp_eval
from cmn.utils import crossValidate

fax = './x_sampleset.pkl'
fay = './y_sampleset.pkl'
with open(fax, 'rb') as f:
    x = pkl.load(f)
with open(fay, 'rb') as f:
    y = pkl.load(f)

# Variables
seed = 7
np.random.seed(seed)
train_ratio = 0.8
validation_ratio = 0.0
epochs = 200
batch_sizes = 8
sp = 0.01
b_val = 3  # Controls the acitvity of the hidden layer nodes
encoding_dim = 3500

# from keras.callbacks import ModelCheckpoint
# from keras.models import Model, load_model, save_model, Sequential
# from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
# from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
# from keras.optimizers import Adam
# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras


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

encoded = Dense(encoding_dim,
                activation='sigmoid')(input_img)

decoded = Dense(output_dim,
                activation='sigmoid')(encoded)


# Define custom loss
def custom_loss(y_true, y_pred):
    # custom function here
    return K.mean((y_pred - y_true), axis=-1)


autoencoder = Model(input_img, decoded)
# sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1, clipvalue=0.5)
# autoencoder.compile(optimizer=sgd, loss=custom_loss)
autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')

# Removing normalization since using MSE - later
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_sizes,
                shuffle=True,
                verbose=2,
                validation_data=(x_validate, y_validate))

print(autoencoder.summary())

evaluation_k_set = np.arange(10, 1100, 100)

# Defining evaluation scores holders for train data
p_at_k_all_train = dblp_eval.init_eval_holder(
    evaluation_k_set)  # all p@k of instances in one fold and one k_evaluation_set
p_at_k_overall_train = dblp_eval.init_eval_holder(
    evaluation_k_set)  # overall p@k of instances in one fold and one k_evaluation_set
r_at_k_all_train = dblp_eval.init_eval_holder(
    evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall_train = dblp_eval.init_eval_holder(
    evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set

# Defining evaluation scores holders for test data
p_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set)  # all p@k of instances in one fold and one k_evaluation_set
p_at_k_overall = dblp_eval.init_eval_holder(
    evaluation_k_set)  # overall p@k of instances in one fold and one k_evaluation_set
r_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall = dblp_eval.init_eval_holder(
    evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set

# @k evaluation process for last train batch data
print("eval on train data.")
for k in evaluation_k_set:
    # p@k evaluation
    print("Evaluating p@k for top {} records.".format(k))
    p_at_k, p_at_k_array = dblp_eval.p_at_k(autoencoder.predict(x_train), y_train, k=k)
    p_at_k_overall_train[k].append(p_at_k)
    p_at_k_all_train[k].append(p_at_k_array)
    # r@k evaluation
    print("Evaluating r@k for top {} records.".format(k))
    r_at_k, r_at_k_array = dblp_eval.r_at_k(autoencoder.predict(x_train), y_train, k=k)
    r_at_k_overall_train[k].append(r_at_k)
    r_at_k_all_train[k].append(r_at_k_array)

    print("For top {} in Train data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))

# @k evaluation process for test data
print("eval on test data.")
for k in evaluation_k_set:
    # p@k evaluation
    print("Evaluating p@k for top {} records.".format(k))
    p_at_k, p_at_k_array = dblp_eval.p_at_k(autoencoder.predict(x_test), y_test, k=k)
    p_at_k_overall[k].append(p_at_k)
    p_at_k_all[k].append(p_at_k_array)
    # r@k evaluation
    print("Evaluating r@k for top {} records.".format(k))
    r_at_k, r_at_k_array = dblp_eval.r_at_k(autoencoder.predict(x_test), y_test, k=k)
    r_at_k_overall[k].append(r_at_k)
    r_at_k_all[k].append(r_at_k_array)

    print("For top {} in Test data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))

# autoencoder.save_weights('./KL.h5')
