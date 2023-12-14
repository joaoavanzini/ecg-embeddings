import pandas as pd

class DataLoader:
    def __init__(self, data_path, num_samples, sample_length):
        self.data_path = data_path
        self.num_samples = num_samples
        self.sample_length = sample_length

    def load_data(self):
        subjects = pd.read_csv(self.data_path)
        subjects = subjects[:self.num_samples * self.sample_length]
        subjects_reshaped = subjects.values.reshape((self.num_samples, self.sample_length, 1))
        return subjects_reshaped
