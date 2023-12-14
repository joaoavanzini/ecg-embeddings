import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Conv1DTranspose, Reshape
import pandas as pd
import numpy as np

class AutoencoderModel:
    def __init__(self, input_shape, conv_filters, conv_kernel_size, dense_units, epochs, batch_size):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None

    def build_model(self):
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            encoder = tf.keras.models.Sequential([
                Conv1D(self.conv_filters, self.conv_kernel_size, activation='relu', padding='same', input_shape=self.input_shape),
                Dense(self.dense_units, activation='relu'),
                MaxPooling1D(),
                Conv1D(self.conv_filters // 2, self.conv_kernel_size, activation='relu', padding='same'),
                Dense(self.dense_units // 4, activation='relu'),
                MaxPooling1D(),
                Flatten(),
                Dense(self.dense_units // 8)
            ], name='encoder')

            decoder = tf.keras.models.Sequential([
                Reshape((self.dense_units // 8, 1)),
                Conv1DTranspose(self.dense_units // 8, (int(self.input_shape[0]) - (self.dense_units // 8) + int(self.input_shape[1])), activation='relu', input_shape=(self.dense_units // 8, 1)),
                Dense(self.dense_units // 4, activation='relu'),
                Conv1DTranspose(self.dense_units // 4, self.conv_kernel_size, activation='relu', padding='same'),
                Dense(self.dense_units // 2, activation='relu'),
                Conv1DTranspose(self.dense_units // 2, self.conv_kernel_size, activation='relu', padding='same'),
                Dense(self.dense_units, activation='relu'),
                Conv1DTranspose(1, self.conv_kernel_size, activation='relu', padding='same')
            ], name='decoder')

            autoencoder = Sequential([encoder, decoder])
            autoencoder.compile(optimizer="adam", loss="mse")
            self.autoencoder = autoencoder

    def get_model(self):
        return self.autoencoder

    def get_epochs(self):
        return self.epochs

    def get_batch_size(self):
        return self.batch_size

    def save_model(self, filepath):
        self.autoencoder.save(filepath)

    def save_metrics(self, metrics_filepath, history):
        rmse_values = np.sqrt(history.history['loss'])
        epochs = range(1, len(rmse_values) + 1)
        metrics_dict = {'epoch': epochs, 'loss': history.history['loss'], 'rmse': rmse_values}
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df.to_csv(metrics_filepath, index=False)