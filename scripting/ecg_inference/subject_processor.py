import pandas as pd
from tensorflow.keras.models import load_model

class SubjectProcessor:
    def __init__(self, model_path, file_path, num_repetitions, sequence_length, num_channels):
        self.model_path = model_path
        self.file_path = file_path
        self.num_repetitions = num_repetitions
        self.sequence_length = sequence_length
        self.num_channels = num_channels

    def process_subject(self):
        print("Loading model...")
        model = load_model(self.model_path)
        model.summary()

        print("Loading subject data...")
        subject_data = pd.read_csv(self.file_path)
        print("Loaded subject_data shape:", subject_data.shape)

        print("Reshaping subject data...")
        subject_data = subject_data.values.reshape((self.num_repetitions, self.sequence_length, self.num_channels))
        print("Reshaped subject_data shape:", subject_data.shape)

        print("Predicting with the model...")
        encoder_embeddings = model.predict(subject_data)
        print("Encoder embeddings shape:", encoder_embeddings.shape)
        
        return encoder_embeddings.flatten()
