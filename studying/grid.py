import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

def create_autoencoder(optimizer='adam'):
    input_layer = Input(shape=(1,))
    encoded = Dense(64, activation='relu')(input_layer)
    decoded = Dense(1, activation='linear')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    return autoencoder

def generate_embeddings_around_time(ecg_data, time_x, sample_rate, embedding_size=32, time_window=2):
    x_index = int(time_x * sample_rate)
    data_x = ecg_data[x_index - time_window * sample_rate : x_index + time_window * sample_rate]
    data_x_reshaped = data_x.reshape(-1, 1)

    return data_x_reshaped

# Generate some example data
duration_minutes = 1
sample_rate = 1000
ecg_signal = nk.ecg_simulate(duration=duration_minutes * 60, noise=0.01, heart_rate=100)

time_x_user = 1 * 60

data_x_reshaped = generate_embeddings_around_time(ecg_signal, time_x_user, sample_rate=sample_rate)

# Split the data into train and test sets
X_train, X_test = train_test_split(data_x_reshaped, test_size=0.3, random_state=42)

# Create the pipeline
pipeline = make_pipeline(StandardScaler(), GradientBoostingRegressor())

# Define the hyperparameters to search
param_grid = {
    'gradientboostingregressor__learning_rate': [0.001, 0.01, 0.1],
    'gradientboostingregressor__n_estimators': [50, 100, 200],
    'gradientboostingregressor__max_depth': [3, 4, 5]
}

# Perform grid search
grid = GridSearchCV(pipeline, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, X_train)

# Print the results
print(f"Best MSE: {grid_result.best_score_} using {grid_result.best_params_}")

# Get the best model
best_model = grid_result.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(X_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(X_test, label='Original Signal')
plt.plot(y_pred, label='Reconstructed Signal')
plt.title('Original vs. Reconstructed Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
