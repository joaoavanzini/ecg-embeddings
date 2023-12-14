import argparse
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt
import numpy as np

from autoencoder_model import AutoencoderModel
from data_loader import DataLoader

class AutoencoderScript:
    def __init__(self, data_path, input_shape, conv_filters, conv_kernel_size, dense_units, epochs, batch_size, num_samples, sample_length):
        self.data_path = data_path
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.autoencoder_model = AutoencoderModel(input_shape, conv_filters, conv_kernel_size, dense_units, epochs, batch_size)
        self.data_loader = DataLoader(data_path, num_samples, sample_length)

    def train_model(self, data):
        history = self.autoencoder_model.get_model().fit(data, data, batch_size=self.autoencoder_model.get_batch_size(),
                                                         epochs=self.autoencoder_model.get_epochs(),
                                                         verbose=0, callbacks=[TqdmCallback(verbose=1)])
        return history

    def plot_loss(self, history):
        plot_df = pd.DataFrame.from_dict({'train_loss': history.history['loss']})
        plot_df.plot(logy=True, figsize=(15, 5), fontsize=12)
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.show()

    def plot_rmse(self, history):
        plt.title("RMSE loss over epochs", fontsize=16)
        plt.plot(np.sqrt(history.history['loss']), label='Training RMSE', c='k', lw=2)
        plt.grid(True)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Root-mean-squared error", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def print_rmse(self, history):
        print("Epoch\tRMSE")
        for epoch, rmse in enumerate(np.sqrt(history.history['loss'])):
            print(f"{epoch}\t{rmse}")

    def run_script(self):
        data = self.data_loader.load_data()
        self.autoencoder_model.build_model()
        history = self.train_model(data)
        self.plot_loss(history)
        self.plot_rmse(history)
        self.print_rmse(history)

        model_filepath = "autoencoder_model.h5"
        self.autoencoder_model.save_model(model_filepath)
        print(f"Model saved at {model_filepath}")

        metrics_filepath = "autoencoder_metrics.csv"
        self.autoencoder_model.save_metrics(metrics_filepath, history)
        print(f"Metrics saved at {metrics_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autoencoder Script')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data CSV file')
    parser.add_argument('--input_shape', type=str, default='30000,1', help='Input shape of the data')
    parser.add_argument('--conv_filters', type=int, default=64, help='Number of convolutional filters')
    parser.add_argument('--conv_kernel_size', type=int, default=1300, help='Convolutional kernel size')
    parser.add_argument('--dense_units', type=int, default=8, help='Number of dense units in the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_samples', type=int, default=1300, help='Number of samples')
    parser.add_argument('--sample_length', type=int, default=30000, help='Length of each sample')

    args = parser.parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))

    autoencoder_script = AutoencoderScript(
        data_path=args.data_path,
        input_shape=input_shape,
        conv_filters=args.conv_filters,
        conv_kernel_size=args.conv_kernel_size,
        dense_units=args.dense_units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        sample_length=args.sample_length
    )
    autoencoder_script.run_script()
