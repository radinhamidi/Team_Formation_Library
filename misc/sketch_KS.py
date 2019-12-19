from misc.k_sparse_autoencoder import KSparse, UpdateSparsityLevel, calculate_sparsity_levels
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import keras
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
batch_sizes = 8
epochs = 10
embedding_size = 3000
initial_sparsity = 1000
sparsity = 15
sparsity_levels = calculate_sparsity_levels(initial_sparsity, sparsity, epochs)

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



encoded = Dense(embedding_size, activation='sigmoid')(input_img)
k_sparse = KSparse(sparsity_levels=sparsity_levels, name='KSparse')(encoded)

decoded = Dense(output_dim, activation='sigmoid')(k_sparse)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adamax', loss='mse')  # What optimizer to use??


# Removing normalization since using MSE - later
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_sizes,
                callbacks=[UpdateSparsityLevel()],
                verbose=2,
                validation_data=(x_validate, y_validate))

print(autoencoder.summary())


evaluation_k_set = np.arange(10, 1100, 100)

# Defining evaluation scores holders for train data
p_at_k_all_train = dblp_eval.init_eval_holder(evaluation_k_set) # all p@k of instances in one fold and one k_evaluation_set
p_at_k_overall_train = dblp_eval.init_eval_holder(evaluation_k_set) # overall p@k of instances in one fold and one k_evaluation_set
r_at_k_all_train = dblp_eval.init_eval_holder(evaluation_k_set) # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall_train = dblp_eval.init_eval_holder(evaluation_k_set) # overall r@k of instances in one fold and one k_evaluation_set

# Defining evaluation scores holders for test data
p_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set) # all p@k of instances in one fold and one k_evaluation_set
p_at_k_overall = dblp_eval.init_eval_holder(evaluation_k_set) # overall p@k of instances in one fold and one k_evaluation_set
r_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set) # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall = dblp_eval.init_eval_holder(evaluation_k_set) # overall r@k of instances in one fold and one k_evaluation_set

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

# autoencoder.save_weights('./KS.h5')