import DataAccessLayer.load_dblp_data as dblp
import Evaluation.Evaluator as dblp_eval
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import KFold
import numpy as np
import time
from keras.utils import plot_model
from contextlib import redirect_stdout
from keras import regularizers

######## Definitions
dataset_name = 'DBLP'
seed = 7
epoch = 5
back_propagation_batch_size = 128
k_fold = 10
evaluation_k_set = np.arange(10, 1100, 100)
training_batch_size = 3000
data_size_limit = 1000
train_ratio = 0.7
validation_ratio = 0.15
skill_size_filter = 0
member_size_filter = 0
encoding_dim = 3000  # encoded size
########

# fix random seed for reproducibility
np.random.seed(seed)
print(K.tensorflow_backend._get_available_gpus())

if dblp.ae_data_exist():
    dataset = dblp.load_ae_dataset()
else:
    dblp.extract_data(filter_journals=True, skill_size_filter=skill_size_filter, member_size_filter=member_size_filter)
    # extract_data(filter_journals=True, size_limit=data_size_limit)
    dataset = dblp.load_ae_dataset()

ids = []
x = []
y = []
for record in dataset:
    ids.append(record[0])
    x.append(record[1])
    y.append(record[2])
del dataset

x = np.asarray(x)
y = np.asarray(y)

# Train/Validation/Test version
# x_train, x_validate, x_test, ids = crossValidate(x, train_ratio, validation_ratio)
# y_train = y[ids[0:int(y.__len__() * train_ratio)]]
# y_validate = y[ids[int(y.__len__() * train_ratio):int(y.__len__() * (train_ratio + validation_ratio))]]
# y_test = y[ids[int(y.__len__() * (train_ratio + validation_ratio)):]]

# 10-fold Cross Validation
cv = KFold(n_splits=k_fold, random_state=seed, shuffle=True)
cvscores = []

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


fold_counter = 1
time_str = time.strftime("%Y%m%d-%H%M%S")
for train_index, test_index in cv.split(x):
    print('Fold number {}'.format(fold_counter))
    x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
    print('Train Size: {} Test Size: {}'.format(x_train.__len__(), x_test.__len__()))
    # this is the size of our encoded representations
    # encoder_hidden_layer_dim = 3675  # encoder first hidden layer
    # encoder_hidden_layer_dim_2 = 4200  # encoder second hidden layer
    # encoder_hidden_layer_dim_3 = 2700  # encoder third hidden layer
    # decoder_hidden_layer_dim = 3675  # decoder first hidden layer
    # data_dim is input dimension
    input_dim = x_train[0].shape[1]
    output_dim = y_train[0].shape[1]
    # this is our input placeholder
    input_img = Input(shape=(input_dim,))
    # hidden #1 in encoder
    # hidden_layer_1 = Dense(encoder_hidden_layer_dim, activation='relu')(input_img)
    # hidden #2 in encoder
    # hidden_layer_1_2 = Dense(encoder_hidden_layer_dim_2, activation='relu')(hidden_layer_1)
    # hidden #3 in encoder
    # hidden_layer_1_3 = Dense(encoder_hidden_layer_dim_2, activation='relu')(hidden_layer_1_2)
    # "encoded" is the encoded representation of the input
    # encoded = Dense(encoding_dim, activation='relu')(hidden_layer_1)
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-4))(input_img)
    # hidden #2 in decoder
    # hidden_layer_2 = Dense(decoder_hidden_layer_dim, activation='relu')(encoded)
    # "decoded" is the lossy reconstruction of the input
    # decoded = Dense(output_dim, activation='sigmoid')(hidden_layer_2)
    decoded = Dense(output_dim, activation='relu', activity_regularizer=regularizers.l1(10e-4))(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(inputs=input_img, outputs=decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    # encoded_input = Input(shape=(encoding_dim,))
    # retrieve hidden layer
    # decoder_hidden = autoencoder.layers[-2]
    # retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    # decoder = Model(encoded_input, decoder_layer(decoder_hidden(encoded_input)))
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adaGrad', loss='mean_absolute_error')

    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # print(x_train.shape)
    # print(x_test.shape)

    # batch_sizes = [128, 64, 32, 4, 1]
    # batch_sizes = [8]
    # epochs = [20]
    # epochs      = [50, 30, 20, 10, 3]
    # for epoch, batch_size in zip(epochs, batch_sizes):
    #     print('*'*30)
    #     print('*'*30)
    print('Starting with back propagation batch size {} for {} times with training batch size {}'
          .format(back_propagation_batch_size, epoch, training_batch_size))
    # Training
    batch_counter = 1
    for x_train_batch, y_train_batch in zip(dblp.batch_generator(x_train, training_batch_size),
                                            dblp.batch_generator(y_train, training_batch_size)):
        print('Training batch {} of fold {}.'.format(batch_counter, fold_counter))
        autoencoder.fit(x_train_batch, y_train_batch,
                        epochs=epoch,
                        batch_size=back_propagation_batch_size,
                        shuffle=True,
                        verbose=2)
        # validation_data=(x_test, y_test))
        batch_counter += 1
        # Evaluating
        # encoded_imgs = encoder.predict(x_test)
        # decoded_imgs = decoder.predict(encoded_imgs)
        # do a loss compute for model:
        # loses = K.eval(mean_squared_error(x_test, decoded_imgs))
        # lose = np.mean(loses)
        # print('Total loss of model is: {0:.4f}'.format(lose))
        # Cool down GPU
        # time.sleep(300)
    score = autoencoder.evaluate(
        np.asarray([x_test_record.todense() for x_test_record in x_test]).reshape(x_test.__len__(), -1),
        np.asarray([y_test_record.todense() for y_test_record in y_test]).reshape(y_test.__len__(), -1),
        verbose=2)
    print('Test loss of fold {}: {}'.format(fold_counter, score))
    cvscores.append(score)

    # @k evaluation process for last train batch data
    print("Evaluation on last batch of train data.")
    for k in evaluation_k_set:
        # p@k evaluation
        print("Evaluating p@k for top {} records in fold {}.".format(k, fold_counter))
        p_at_k, p_at_k_array = dblp_eval.p_at_k(autoencoder.predict(x_train_batch), y_train_batch, k=k)
        p_at_k_overall_train[k].append(p_at_k)
        p_at_k_all_train[k].append(p_at_k_array)
        # r@k evaluation
        print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
        r_at_k, r_at_k_array = dblp_eval.r_at_k(autoencoder.predict(x_train_batch), y_train_batch, k=k)
        r_at_k_overall_train[k].append(r_at_k)
        r_at_k_all_train[k].append(r_at_k_array)

        print("For top {} in Train data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))

    # @k evaluation process for test data
    print("Evaluation on test data.")
    for k in evaluation_k_set:
        # p@k evaluation
        print("Evaluating p@k for top {} records in fold {}.".format(k, fold_counter))
        p_at_k, p_at_k_array = dblp_eval.p_at_k(autoencoder.predict(
            np.asarray([x_test_record.todense() for x_test_record in x_test]).reshape(x_test.__len__(), -1)),
            np.asarray([y_test_record.todense() for y_test_record in y_test]).reshape(y_test.__len__(), -1), k=k)
        p_at_k_overall[k].append(p_at_k)
        p_at_k_all[k].append(p_at_k_array)
        # r@k evaluation
        print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
        r_at_k, r_at_k_array = dblp_eval.r_at_k(autoencoder.predict(
            np.asarray([x_test_record.todense() for x_test_record in x_test]).reshape(x_test.__len__(), -1)),
            np.asarray([y_test_record.todense() for y_test_record in y_test]).reshape(y_test.__len__(), -1), k=k)
        r_at_k_overall[k].append(r_at_k)
        r_at_k_all[k].append(r_at_k_array)

        print("For top {} in Test data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))

    # for test_instance in x_test:
    #     result = autoencoder.predict(test_instance)

    # saving model
    # save_model_q = input('Save the models? (y/n)')
    # if save_model_q.lower() == 'y':
    # encoder_model_json = encoder.to_json()
    # decoder_model_json = decoder.to_json()
    model_json = autoencoder.to_json()

    # encoder_name = input('Please enter encoder model name:')
    # decoder_name = input('Please enter decoder model name:')
    # model_name = input('Please enter autoencoder model name:')

    # with open('./Models/{}.json'.format(encoder_name), "w") as json_file:
    #     json_file.write(encoder_model_json)
    # with open('./Models/{}.json'.format(decoder_name), "w") as json_file:
    #     json_file.write(decoder_model_json)
    with open('../Output/Models/{}_Time{}_Fold{}.json'.format(dataset_name, time_str, fold_counter), "w") as json_file:
        json_file.write(model_json)

    # encoder.save_weights("./Models/weights/{}.h5".format(encoder_name))
    # decoder.save_weights("./Models/weights/{}.h5".format(decoder_name))
    autoencoder.save_weights("../Output/Models/Weights/{}_Time{}_Fold{}.h5".format(dataset_name, time_str, fold_counter))

    with open('../Output/Models/{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}_kFold{}_BatchBP{}_BatchTraining{}.txt'
                      .format(dataset_name, time_str, encoding_dim, fold_counter, int(np.mean(cvscores) * 1000), epoch, k_fold,
                              back_propagation_batch_size, training_batch_size), 'w') as f:
        with redirect_stdout(f):
            autoencoder.summary()

    # plot_model(autoencoder, '../Output/Models/{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}_kFold{}_BatchBP{}_BatchTraining{}.png'
    #            .format(dataset_name, time_str, encoding_dim, fold_counter, int(np.mean(cvscores) * 1000), epoch, k_fold,
    #                    back_propagation_batch_size, training_batch_size))
    # print('Model and its summary and architecture plot are saved.')
    print('Model and its summary are saved.')

    # Deleting model from RAM
    K.clear_session()

    # Saving evaluation data
    dblp_eval.save_record(p_at_k_all_train, '{}_p@k_all_train_Time{}'.format(dataset_name, time_str))
    dblp_eval.save_record(p_at_k_overall_train, '{}_p@k_train_Time{}'.format(dataset_name, time_str))
    dblp_eval.save_record(r_at_k_all_train, '{}_r@k_all_train_Time{}'.format(dataset_name, time_str))
    dblp_eval.save_record(p_at_k_overall_train, '{}_r@k_train_Time{}'.format(dataset_name, time_str))

    dblp_eval.save_record(p_at_k_all, '{}_p@k_all_Time{}'.format(dataset_name, time_str))
    dblp_eval.save_record(p_at_k_overall, '{}_p@k_Time{}'.format(dataset_name, time_str))
    dblp_eval.save_record(r_at_k_all, '{}_r@k_all_Time{}'.format(dataset_name, time_str))
    dblp_eval.save_record(p_at_k_overall, '{}_r@k_Time{}'.format(dataset_name, time_str))

    print('Evaluation records are saved successfully.')

    fold_counter += 1
