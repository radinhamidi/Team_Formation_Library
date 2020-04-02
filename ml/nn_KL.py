import numpy as np
import time
import glob
import pickle as pkl
import csv
from contextlib import redirect_stdout
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras import regularizers
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
import cmn.utils
import dal.load_dblp_data as dblp
import eval.evaluator as dblp_eval
from cmn.utils import *
from ml.nn_custom_func import *
import eval.ranking as rk
import ml_metrics as metrics

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, min_delta=1)

#running settings
dataset_name = 'DBLP'
method_name = 'O_KL_O'

#eval settings
k_fold = 10
k_max = 100
evaluation_k_set = np.arange(1, k_max+1, 1)

#nn settings
epochs_in_batch = 25
epochs_overall = 10
back_propagation_batch_size = 64
training_batch_size = 6000
min_skill_size = 0
min_member_size = 0
encoding_dim = 700


print(K.tensorflow_backend._get_available_gpus())

if dblp.preprocessed_dataset_exist() and dblp.train_test_indices_exist():
    dataset = dblp.load_preprocessed_dataset()
    train_test_indices = dblp.load_train_test_indices()
else:
    if not dblp.ae_data_exist(file_path='../dataset/ae_dataset.pkl'):
        dblp.extract_data(filter_journals=True, skill_size_filter=min_skill_size, member_size_filter=min_member_size)
    if not dblp.preprocessed_dataset_exist() or not dblp.train_test_indices_exist():
        dblp.dataset_preprocessing(dblp.load_ae_dataset(file_path='../dataset/ae_dataset.pkl'), seed=seed, kfolds=k_fold)
    dataset = dblp.load_preprocessed_dataset()
    train_test_indices = dblp.load_train_test_indices()

# k_fold Cross Validation
cvscores = []

# Defining evaluation scores holders for train data
r_at_k_all_train = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall_train = dblp_eval.init_eval_holder(evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set

# Defining evaluation scores holders for test data
r_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall = dblp_eval.init_eval_holder(evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set
mapk = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
ndcg = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
mrr = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set

lambda_val = 0.001  # Weight decay , refer : https://stackoverflow.com/questions/44495698/keras-difference-between-kernel-and-activity-regularizers

load_weights_from_file_q = input('Load weights from file? (y/n)')
more_train_q = input('Train more? (y/n)')

time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
result_output_name = "../output/predictions/{}_output.csv".format(method_name)
with open(result_output_name, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Method Name', '# Total Folds', '# Fold Number', '# Predictions', '# Truth', 'Computation Time (ms)',
         'Prediction Indices', 'True Indices'])

for fold_counter in range(1,k_fold+1):
    x_train, y_train, x_test, y_test = dblp.get_fold_data(fold_counter, dataset, train_test_indices)

    input_dim = x_train[0][0].shape[1]
    output_dim = y_train[0][0].shape[1]
    print("Input/output Dimensions:  ", input_dim, output_dim)

    # this is our input placeholder
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_val / 2), activity_regularizer=sparse_reg)(input_img)
    decoded = Dense(output_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_val / 2), activity_regularizer=sparse_reg)(encoded)
    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='adagrad', loss='mean_absolute_error')
    # autoencoder.compile(optimizer='adagrad', loss='cross_entropy')

    # Loading model weights
    if load_weights_from_file_q.lower() == 'y':
        pick_model_weights(autoencoder, dataset_name=dataset_name)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    if more_train_q.lower() == 'y':
        print('Starting with back propagation batch size {} for {} times each during {} overall epochs with training batch size {}. Total epochs={}.'
            .format(back_propagation_batch_size,
                    epochs_in_batch,
                    epochs_overall,
                    training_batch_size,
                    epochs_overall * epochs_in_batch))
        # Training
        for epoch_overall in range(epochs_overall):
            batch_counter = 1
            for x_train_batch, y_train_batch in zip(batch_generator(x_train, training_batch_size), batch_generator(y_train, training_batch_size)):
                print('Overall Epoch: {} | Training batch {} of fold {}.'.format(epoch_overall + 1, batch_counter, fold_counter))
                autoencoder.fit(x_train_batch, y_train_batch,
                                epochs=epochs_in_batch,
                                batch_size=back_propagation_batch_size,
                                shuffle=True,
                                callbacks=[es],
                                verbose=2,
                                validation_data=(
                                np.asarray([x_test_record[0].todense() for x_test_record in x_test]).reshape(x_test.__len__(),-1),
                                np.asarray([y_test_record[0].todense() for y_test_record in y_test]).reshape(y_test.__len__(),-1)))
                batch_counter += 1
                # Cool down GPU
                # time.sleep(300)

    score = autoencoder.evaluate(
        np.asarray([x_test_record[0].todense() for x_test_record in x_test]).reshape(x_test.__len__(), -1),
        np.asarray([y_test_record[0].todense() for y_test_record in y_test]).reshape(y_test.__len__(), -1),
        verbose=2)
    print('Test loss of fold {}: {}'.format(fold_counter, score))
    cvscores.append(score)

    # # @k evaluation process for last train batch data
    # print("eval on last batch of train data.")
    # for k in evaluation_k_set:
    #     # r@k evaluation
    #     print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
    #     r_at_k, r_at_k_array = dblp_eval.r_at_k(autoencoder.predict(x_train_batch), y_train_batch, k=k)
    #     r_at_k_overall_train[k].append(r_at_k)
    #     r_at_k_all_train[k].append(r_at_k_array)
    #
    #     # print("For top {} in Train data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))
    #     print("For top {} in train data: R@{}:{}".format(k, k, r_at_k))

    # @k evaluation process for test data
    print("eval on test data fold #{}".format(fold_counter))
    true_indices = []
    pred_indices = []
    x_test = np.asarray([x_test_record[0].todense() for x_test_record in x_test]).reshape(x_test.__len__(), -1)
    y_test = np.asarray([y_test_record[0].todense() for y_test_record in y_test]).reshape(y_test.__len__(), -1)
    with open(result_output_name, 'a+') as file:
        writer = csv.writer(file)
        for sample_x, sample_y in zip(x_test, y_test):
            start_time = time.time()
            sample_prediction = autoencoder.predict(np.asmatrix(sample_x))
            end_time = time.time()
            elapsed_time = (end_time - start_time)*1000
            pred_index, true_index = dblp_eval.find_indices(sample_prediction, [sample_y])
            true_indices.append(true_index[0])
            pred_indices.append(pred_index[0])
            writer.writerow([method_name, k_fold, fold_counter, len(pred_index[0][:k_max]), len(true_index[0]),
                             elapsed_time] + pred_index[0][:k_max] + true_index[0])

    # print("eval on test data.")
    # prediction_test = autoencoder.predict(
    #         np.asarray([x_test_record[0].todense() for x_test_record in x_test]).reshape(x_test.__len__(), -1))
    # y_test = np.asarray([y_test_record[0].todense() for y_test_record in y_test]).reshape(y_test.__len__(), -1)
    # pred_indices, true_indices = dblp_eval.find_indices(prediction_test, y_test)
    # for k in evaluation_k_set:
    #     # r@k evaluation
    #     print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
    #     r_at_k, r_at_k_array = dblp_eval.r_at_k(pred_indices, true_indices, k=k)
    #     r_at_k_overall[k].append(r_at_k)
    #     r_at_k_all[k].append(r_at_k_array)
    #     mapk[k].append(metrics.mapk(true_indices, pred_indices, k=k))
    #     ndcg[k].append(rk.ndcg_at(pred_indices, true_indices, k=k))
    #     # print("For top {} in Test data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))
    #     print("For top {} in test data: R@{}:{}".format(k, k, r_at_k))
    #     print("For top {} in test data: MAP@{}:{}".format(k, k, mapk[k][-1]))
    #     print("For top {} in test data: NDCG@{}:{}".format(k, k, ndcg[k][-1]))
    # mrr[k].append(dblp_eval.mean_reciprocal_rank(dblp_eval.cal_relevance_score(pred_indices, true_indices)))


    # saving model
    # save_model_q = input('Save the models? (y/n)')
    # if save_model_q.lower() == 'y':
    model_json = autoencoder.to_json()

    # model_name = input('Please enter autoencoder model name:')

    with open('../output/Models/{}_{}_Time{}_Fold{}.json'.format(dataset_name, method_name, time_str, fold_counter), "w") as json_file:
        json_file.write(model_json)

    autoencoder.save_weights(
        "../output/Models/Weights/{}_{}_Time{}_Fold{}.h5".format(dataset_name, method_name, time_str, fold_counter))

    with open('../output/Models/{}_{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}(overall{}xinner{})_kFold{}_BatchBP{}_BatchTraining{}.txt'
                    .format(dataset_name, method_name, time_str, encoding_dim, fold_counter, int(score * 1000),
                            epochs_in_batch * epochs_overall, epochs_in_batch, epochs_overall,
                            k_fold, back_propagation_batch_size, training_batch_size), 'w') as f:
        with redirect_stdout(f):
            autoencoder.summary()

    # plot_model(autoencoder, '../output/Models/{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}_kFold{}_BatchBP{}_BatchTraining{}.png'
    #            .format(dataset_name, time_str, encoding_dim, fold_counter, int(np.mean(cvscores) * 1000), epoch, k_fold,
    #                    back_propagation_batch_size, training_batch_size))
    # print('Model and its summary and architecture plot are saved.')
    print('Model and its summary are saved.')

    # Deleting model from RAM
    K.clear_session()

    # Saving evaluation data
    # cmn.utils.save_record(r_at_k_all_train, '{}_{}_r@k_all_train_Time{}'.format(dataset_name, method_name, time_str))
    # cmn.utils.save_record(r_at_k_overall_train, '{}_{}_r@k_train_Time{}'.format(dataset_name, method_name, time_str))
    #
    # cmn.utils.save_record(r_at_k_all, '{}_{}_r@k_all_Time{}'.format(dataset_name, method_name, time_str))
    # cmn.utils.save_record(r_at_k_overall, '{}_{}_r@k_Time{}'.format(dataset_name, method_name, time_str))
    #
    # print('eval records are saved successfully for fold #{}'.format(fold_counter))

    fold_counter += 1
    # break

print('Loss for each fold: {}'.format(cvscores))

# compare_submit = input('Submit for compare? (y/n)')
# if compare_submit.lower() == 'y':
#     with open('../misc/{}_r_at_k_50.pkl'.format(method_name), 'wb') as f:
#         pkl.dump(r_at_k_overall, f)
#     with open('../misc/{}_mapk_50.pkl'.format(method_name), 'wb') as f:
#         pkl.dump(mapk, f)
#     with open('../misc/{}_ndcg_50.pkl'.format(method_name), 'wb') as f:
#         pkl.dump(ndcg, f)
#     with open('../misc/{}_mrr_50.pkl'.format(method_name), 'wb') as f:
#         pkl.dump(mrr, f)