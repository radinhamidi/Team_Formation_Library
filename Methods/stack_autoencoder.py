import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from keras.models import model_from_json
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mean_squared_error

from keras import backend as K

print(K.tensorflow_backend._get_available_gpus())
seed = 7
np.random.seed(seed)

# load data
train = np.load('./dataset/train_x_50.npy')
test = np.load('./dataset/test_x_50.npy')
labels = np.loadtxt('./dataset/class_names.txt', dtype=np.str)
print('Data loaded.')

# load encoder model
model_names = []
for path in glob.glob('./models/encoder_*.json'):
    model_names.append(path)
print('Please enter you model number form list below:')
for i, path in enumerate(model_names):
    print('{}. {}'.format(i, path))
model_number = int(input('?'))
encoder_model_name = model_names[model_number].replace('.json', '')
decoder_model_name = encoder_model_name.replace('encoder', 'decoder')
encoder_weight_name = encoder_model_name.replace('models', 'weights')
decoder_weight_name = decoder_model_name.replace('models', 'weights')

encoder_json_file = open('{}.json'.format(encoder_model_name), 'r')
loaded_encoder_json = encoder_json_file.read()
encoder_json_file.close()
encoder_1 = model_from_json(loaded_encoder_json)

decoder_json_file = open('{}.json'.format(decoder_model_name), 'r')
loaded_decoder_json = decoder_json_file.read()
decoder_json_file.close()
decoder_1 = model_from_json(loaded_decoder_json)

# load weights into new model
encoder_1.load_weights("{}.h5".format(encoder_weight_name))
decoder_1.load_weights("{}.h5".format(decoder_weight_name))
print("Loaded: {} model from disk".format(encoder_model_name.replace('./models\\', '')))
print("Loaded: {} model from disk".format(decoder_model_name.replace('./models\\', '')))

# compile models
encoder_1.compile(optimizer='adadelta', loss='mean_squared_error')
decoder_1.compile(optimizer='adadelta', loss='mean_squared_error')

# featurize images
train = train.astype('float32') / 255.
test = test.astype('float32') / 255.
raw_test = test # for total model evaluation step
train_encoded = encoder_1.predict(train)
test_encoded = encoder_1.predict(test)
print('Images converted to features via encoder.')

# this is the size of our encoded representations
encoder_hidden_layer_dim = 1100  # first hidden layer dim
encoder_hidden_layer_dim_2 = 2700  # second hidden layer dim
encoding_dim = 1000  # input dim
decoder_hidden_layer_dim = 1100  # first hidden layer in decoder
# data_dim is input dimension
data_dim = train_encoded.shape[1]
# this is our input placeholder
input_img = Input(shape=(data_dim,))
# hidden #1 in encoder
hidden_layer_1 = Dense(encoder_hidden_layer_dim, activation='relu')(input_img)
# hidden #2 in encoder
# hidden_layer_1_2 = Dense(encoder_hidden_layer_dim_2, activation='relu')(hidden_layer_1)
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(hidden_layer_1)
# encoded = Dense(encoding_dim, activation='relu')(input_img)
# hidden #2 in decoder
hidden_layer_decoder = Dense(decoder_hidden_layer_dim, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(data_dim, activation='sigmoid')(hidden_layer_decoder)
# decoded = Dense(data_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded, name='autoencoder')

# this model maps an input to its encoded representation
encoder_2 = Model(input_img, encoded, name='encoder_2')

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve hidden layer
decoder_hidden = autoencoder.layers[-2]
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder_2 = Model(encoded_input, decoder_layer(decoder_hidden(encoded_input)), name='decoder_2')
# decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

x_train = train_encoded.reshape((len(train_encoded), np.prod(train_encoded.shape[1:])))
x_test = test_encoded.reshape((len(test_encoded), np.prod(test_encoded.shape[1:])))
print(x_train.shape)
print(x_test.shape)

batch_sizes = [128, 64, 32, 4, 1]
epochs      = [50, 30, 20, 10, 3]
for epoch, batch_size in zip(epochs, batch_sizes):
    print('*'*30)
    print('*'*30)
    print('Starting batch size {} for {} times.'.format(batch_size, epoch))
    # Training
    autoencoder.fit(x_train, x_train,
                    epochs=epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=2,
                    validation_data=(x_test, x_test))

    # Evaluating
    encoded_imgs = encoder_2.predict(x_test)
    decoded_imgs = decoder_2.predict(encoded_imgs)
    final_imgs = decoder_1.predict(decoded_imgs)
    # do a loss compute for stacked model:
    loses = K.eval(mean_squared_error(raw_test, final_imgs))
    lose = np.mean(loses)
    print('Total loss of stack model is: {0:.4f}'.format(lose))
    # Cool down GPU
    time.sleep(600)

# creating stacked models
encoder_stacked = Model(encoder_1.input, encoder_2(encoder_1.output), name='encoder_stacked')

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test[i].reshape(50, 50, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(final_imgs[i].reshape(50, 50, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# saving model
save_model_q = input('Save the stacked encoder model? (y/n)')
if save_model_q == 'y':
    encoder_model_json = encoder_stacked.to_json()

    encoder_name = input('Please enter encoder model name:')

    with open('./models/{}.json'.format(encoder_name), "w") as json_file:
        json_file.write(encoder_model_json)
    encoder_stacked.save_weights("./weights/{}.h5".format(encoder_name))
