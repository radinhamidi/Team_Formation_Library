import pickle as pkl
import time
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from contextlib import redirect_stdout
import csv
import cmn.utils
from keras.callbacks import EarlyStopping
from cmn.utils import *
import dal.load_dblp_data as dblp
import eval.evaluator as dblp_eval
from ml.nn_custom_func import *
import eval.ranking as rk
import ml_metrics as metrics

# fix random seed for reproducibility
seed = 10
np.random.seed(seed)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=1)

#settings
dataset_name = 'DBLP'
method_name = 'O_KL_U'

#eval settings
k_fold = 10
k_max = 100
evaluation_k_set = np.arange(1, k_max+1, 1)

#nn settings
epochs = 200
back_propagation_batch_size = 32
min_skill_size = 0
min_member_size = 0
encoding_dim = 100

print(K.tensorflow_backend._get_available_gpus())

t2v_model = Team2Vec()
t2v_model = load_T2V_model(t2v_model)
embedding_dim = t2v_model.model.vector_size

if dblp.ae_data_exist(file_path='../dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim)):
    dataset = dblp.load_ae_dataset(file_path='../dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim))
else:
    if not dblp.ae_data_exist(file_path='../dataset/ae_dataset.pkl'):
        dblp.extract_data(filter_journals=True, skill_size_filter=min_skill_size, member_size_filter=min_member_size)
    if not dblp.preprocessed_dataset_exist() or not dblp.train_test_indices_exist():
        dblp.dataset_preprocessing(dblp.load_ae_dataset(file_path='../dataset/ae_dataset.pkl'), seed=seed, kfolds=k_fold, shuffle_at_the_end=True)
    preprocessed_dataset = dblp.load_preprocessed_dataset()

    dblp.nn_t2v_dataset_generator(t2v_model, preprocessed_dataset, output_file_path='../dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim), mode='user')
    del preprocessed_dataset
    dataset = dblp.load_ae_dataset(file_path='../dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim))



# k-fold Cross Validation
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
result_output_name = "../output/predictions/{}_{}.csv".format(method_name, time_str)
with open(result_output_name, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Method Name', '# Total Folds', '# Fold Number', '# Predictions', '# Truth', 'Computation Time (ms)',
         'Prediction Indices', 'True Indices'])

train_test_indices = dblp.load_train_test_indices()
for fold_counter in range(1,k_fold+1):
    x_train, y_train, x_test, y_test = dblp.get_fold_data(fold_counter, dataset, train_test_indices)

    train_index = train_test_indices[fold_counter]['Train']
    test_index = train_test_indices[fold_counter]['Test']
    y_sparse_train = []
    y_sparse_test = []
    preprocessed_dataset = dblp.load_preprocessed_dataset()
    for sample in preprocessed_dataset:
        id = sample[0]
        if id in train_index:
            y_sparse_train.append(sample[2])
        elif id in test_index:
            y_sparse_test.append(sample[2])
    y_sparse_train = np.asarray(y_sparse_train).reshape(y_sparse_train.__len__(), -1)
    y_sparse_test = np.asarray(y_sparse_test).reshape(y_sparse_test.__len__(), -1)
    del preprocessed_dataset

    input_dim = x_train[0].shape[0]
    output_dim = y_train[0].shape[0]
    print("Input/output Dimensions:  ", input_dim, output_dim)
    # this is our input placeholder
    input_img = Input(shape=(input_dim,))

    encoded = Dense(encoding_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_val / 2), activity_regularizer=sparse_reg)(input_img)
    decoded = Dense(output_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_val / 2), activity_regularizer=sparse_reg)(encoded)
    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='adagrad', loss='mse')

    # Loading model weights
    if load_weights_from_file_q.lower() == 'y':
        pick_model_weights(autoencoder, dataset_name=dataset_name)

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    if more_train_q.lower() == 'y':
        # Training
        autoencoder.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=back_propagation_batch_size,
                        callbacks=[es],
                        shuffle=True,
                        verbose=2,
                        validation_data=(x_test, y_test))

    score = autoencoder.evaluate(x_test, y_test)
    print('Test loss of fold {}: {}'.format(fold_counter, score))
    cvscores.append(score)

    # Team mode evaluation
    # y_train_pred = [[int(candidate[0]) for candidate in t2v_model.get_team_most_similar_by_vector(record, k_max)]
    #           for record in autoencoder.predict(x_train)]
    # y_train_pred = dblp.get_memebrID_by_teamID(y_train_pred)
    #
    # y_test_pred = [[int(candidate[0]) for candidate in t2v_model.get_team_most_similar_by_vector(record, k_max)]
    #           for record in autoencoder.predict(x_test)]
    # y_test_pred = dblp.get_memebrID_by_teamID(y_test_pred)


    # Member mode evaluation
    # y_train_pred = [[int(candidate[0]) for candidate in t2v_model.get_member_most_similar_by_vector(record, k_max)]
    #           for record in autoencoder.predict(x_train)]
    # y_test_pred = [[int(candidate[0]) for candidate in t2v_model.get_member_most_similar_by_vector(record, k_max)]
    #           for record in autoencoder.predict(x_test)]

    # # @k evaluation process for last train batch data
    # print("eval on last batch of train data.")
    # for k in evaluation_k_set:
    #     print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
    #     r_at_k, r_at_k_array = dblp_eval.r_at_k_t2v(y_train_pred, y_sparse_train, k=k)
    #     r_at_k_overall_train[k].append(r_at_k)
    #     r_at_k_all_train[k].append(r_at_k_array)
    #     print("For top {} in Train data: R@{}:{}".format(k, k, r_at_k))

    # @k evaluation process for test data
    print("eval on test data fold #{}".format(fold_counter))
    true_indices = []
    pred_indices = []
    with open(result_output_name, 'a+') as file:
        writer = csv.writer(file)
        for sample_x, sample_y in zip(x_test, y_sparse_test):
            start_time = time.time()
            record = autoencoder.predict(np.asmatrix(sample_x))
            end_time = time.time()
            sample_prediction = [[int(candidate[0]) for candidate in t2v_model.get_member_most_similar_by_vector(record[0], k_max)]]
            elapsed_time = (end_time - start_time)*1000
            pred_index, true_index = dblp_eval.find_indices_t2v(sample_prediction, [sample_y])
            true_indices.append(true_index[0])
            pred_indices.append(pred_index[0])
            writer.writerow([method_name, k_fold, fold_counter, len(pred_index[0][:k_max]), len(true_index[0]),
                             elapsed_time] + pred_index[0][:k_max] + true_index[0])

    # pred_indices, true_indices = dblp_eval.find_indices_t2v(y_test_pred, y_sparse_test)
    # print("eval on test data.")
    # for k in evaluation_k_set:
    #     # r@k evaluation
    #     print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
    #     r_at_k, r_at_k_array = dblp_eval.r_at_k(pred_indices, true_indices, k=k)
    #     r_at_k_overall[k].append(r_at_k)
    #     r_at_k_all[k].append(r_at_k_array)
    #     mapk[k].append(metrics.mapk(true_indices, pred_indices, k=k))
    #     ndcg[k].append(rk.ndcg_at(pred_indices, true_indices, k=k))
    #     print("For top {} in test data: R@{}:{}".format(k, k, r_at_k))
    #     print("For top {} in test data: MAP@{}:{}".format(k, k, mapk[k][-1]))
    #     print("For top {} in test data: NDCG@{}:{}".format(k, k, ndcg[k][-1]))
    # mrr[k].append(dblp_eval.mean_reciprocal_rank(dblp_eval.cal_relevance_score(pred_indices, true_indices)))
    # print("For top {} in test data: MRR@{}:{}".format(k, k, mrr[k][-1]))


    # for test_instance in x_test:
    #     result = autoencoder.predict(test_instance)

    # saving model
    # save_model_q = input('Save the models? (y/n)')
    # if save_model_q.lower() == 'y':
    model_json = autoencoder.to_json()

    # model_name = input('Please enter autoencoder model name:')

    with open('../output/Models/{}_{}_Time{}_Fold{}.json'.format(dataset_name, method_name, time_str, fold_counter), "w") as json_file:
        json_file.write(model_json)

    autoencoder.save_weights("../output/Models/Weights/{}_{}_Time{}_Fold{}.h5".format(dataset_name, method_name, time_str, fold_counter))

    with open('../output/Models/{}_{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}_kFold{}_BatchBP{}.txt'
                    .format(dataset_name, method_name, time_str, encoding_dim, fold_counter, int(score * 1000),
                            epochs, k_fold, back_propagation_batch_size), 'w') as f:
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
#     with open('../misc/{}_dim{}_r_at_k_50.pkl'.format(method_name, embedding_dim), 'wb') as f:
#         pkl.dump(r_at_k_overall, f)
#     with open('../misc/{}_dim{}_mapk_50.pkl'.format(method_name, embedding_dim), 'wb') as f:
#         pkl.dump(mapk, f)
#     with open('../misc/{}_dim{}_ndcg_50.pkl'.format(method_name, embedding_dim), 'wb') as f:
#         pkl.dump(ndcg, f)
#     with open('../misc/{}_dim{}_mrr_50.pkl'.format(method_name, embedding_dim), 'wb') as f:
#         pkl.dump(mrr, f)
