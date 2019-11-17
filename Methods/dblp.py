from DataAccessLayer.load_dblp_data import *
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import KFold
import numpy as np
import time
from keras.utils import plot_model
from contextlib import redirect_stdout

######## Definitions
seed = 7
epoch = 10
back_propagation_batch_size = 8
k_fold = 10
training_batch_size = 300
data_size_limit = 1000
train_ratio = 0.7
validation_ratio = 0.15
########

# fix random seed for reproducibility
np.random.seed(seed)
print(K.tensorflow_backend._get_available_gpus())

if ae_data_exist():
    dataset = load_ae_dataset()
else:
    extract_data(filter_journals=True)
    dataset = load_ae_dataset()

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

fold_counter = 1
for train_index, test_index in cv.split(x):
    print('Fold number {}'.format(fold_counter))
    x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
    # this is the size of our encoded representations
    # encoder_hidden_layer_dim = 3675  # encoder first hidden layer
    # encoder_hidden_layer_dim_2 = 4200  # encoder second hidden layer
    # encoder_hidden_layer_dim_3 = 2700  # encoder third hidden layer
    encoding_dim = 2000  # encoded size
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
    encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
    # hidden #2 in decoder
    # hidden_layer_2 = Dense(decoder_hidden_layer_dim, activation='relu')(encoded)
    # "decoded" is the lossy reconstruction of the input
    # decoded = Dense(output_dim, activation='sigmoid')(hidden_layer_2)
    decoded = Dense(output_dim, activation='sigmoid')(encoded)

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

    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

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
    for x_train_batch, y_train_batch in zip(batch_generator(x_train, training_batch_size),
                                            batch_generator(y_train, training_batch_size)):
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
    score = autoencoder.evaluate([x_test_record.todense() for x_test_record in x_test],
                                 [y_test_record.todense() for y_test_record in y_test], verbose=2)
    print('Test loss of fold {}: {}'.format(fold_counter, score))
    cvscores.append(score)
    fold_counter += 1

# for test_instance in x_test:
#     result = autoencoder.predict(test_instance)

# saving model
save_model_q = input('Save the models? (y/n)')
if save_model_q.lower() == 'y':
    time_str = time.strftime("%Y%m%d-%H%M%S")
    # encoder_model_json = encoder.to_json()
    # decoder_model_json = decoder.to_json()
    model_json = autoencoder.to_json()

    # encoder_name = input('Please enter encoder model name:')
    # decoder_name = input('Please enter decoder model name:')
    model_name = input('Please enter autoencoder model name:')

    # with open('./Models/{}.json'.format(encoder_name), "w") as json_file:
    #     json_file.write(encoder_model_json)
    # with open('./Models/{}.json'.format(decoder_name), "w") as json_file:
    #     json_file.write(decoder_model_json)
    with open('../Output/Models/{}.json'.format(model_name), "w") as json_file:
        json_file.write(model_json)

    # encoder.save_weights("./Models/weights/{}.h5".format(encoder_name))
    # decoder.save_weights("./Models/weights/{}.h5".format(decoder_name))
    autoencoder.save_weights("../Output/Models/Weights/{}.h5".format(model_name))

    with open('../Output/Models/{}_Time{}_ACC{}_Loss{}_Epoch{}_kFold{}_BatchBP{}_BatchTraining{}.txt'
                      .format(model_name, time_str, 0, int(np.mean(cvscores) * 1000), epoch, k_fold,
                              back_propagation_batch_size, training_batch_size), 'w') as f:
        with redirect_stdout(f):
            autoencoder.summary()

    plot_model(autoencoder, '../Output/Models/{}_Time{}_ACC{}_Loss{}_Epoch{}_kFold{}_BatchBP{}_BatchTraining{}.png'
               .format(model_name, time_str, 0, int(np.mean(cvscores) * 1000), epoch, k_fold,
                       back_propagation_batch_size, training_batch_size))
    print('Model and its summary saved.')
