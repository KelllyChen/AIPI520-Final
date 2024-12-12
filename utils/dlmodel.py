import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Function to create sequences from the data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)].values)
        y.append(data.iloc[i + seq_length]['Close.Change.Pct'])
    return np.array(X), np.array(y)

# Function to define and compile the model
def build_model(lstm_units, dropout_rate, learning_rate, input_shape):
    model = Sequential([
        LSTM(lstm_units, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        BatchNormalization(),
        LSTM(lstm_units // 2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model

# Function for hyperparameter tuning
def hyperparameter_tuning(X, y, param_grid, seq_length):
    param_combinations = list(product(
        param_grid['lstm_units'],
        param_grid['dropout_rate'],
        param_grid['learning_rate'],
        param_grid['batch_size']
    ))

    best_params = None
    best_avg_mse = float('inf')

    for lstm_units, dropout_rate, learning_rate, batch_size in param_combinations:
        print(f"Testing combination: LSTM Units={lstm_units}, Dropout={dropout_rate}, Learning Rate={learning_rate}, Batch Size={batch_size}")
        
        fold_mse = []
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train_reshaped = X_train.reshape(X_train.shape[0], seq_length, -1)
            X_test_reshaped = X_test.reshape(X_test.shape[0], seq_length, -1)

            model = build_model(lstm_units, dropout_rate, learning_rate, (seq_length, X_train_reshaped.shape[2]))
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(
                X_train_reshaped, y_train,
                epochs=10,
                batch_size=batch_size,
                validation_data=(X_test_reshaped, y_test),
                verbose=0,
                callbacks=[lr_scheduler, early_stopping]
            )

            predictions = model.predict(X_test_reshaped)
            mse = mean_squared_error(y_test, predictions)
            fold_mse.append(mse)

        avg_mse = np.mean(fold_mse)
        print(f"Average MSE: {avg_mse}")

        if avg_mse < best_avg_mse:
            best_avg_mse = avg_mse
            best_params = {
                'lstm_units': lstm_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }

    print("\nBest Parameters:")
    print(best_params)
    print(f"Best Average MSE: {best_avg_mse}")
    return best_params

# Main function
def main():
    # Set the working directory and load the data
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data', 'sp500_data.csv')
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values('Date').drop(columns=['Date'])

    # Preprocess data
    seq_length = 30
    X, y = create_sequences(df, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Hyperparameter tuning
    param_grid = {
        'lstm_units': [128],
        'dropout_rate': [0.1, 0.2],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [16]
    }
    best_params = hyperparameter_tuning(X, y, param_grid, seq_length)

    # Train final model
    X_train_reshaped = X_train.reshape(X_train.shape[0], seq_length, -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], seq_length, -1)

    final_model = build_model(
        best_params['lstm_units'],
        best_params['dropout_rate'],
        best_params['learning_rate'],
        (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
    )

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = final_model.fit(
        X_train_reshaped, y_train,
        epochs=50,
        batch_size=best_params['batch_size'],
        validation_data=(X_test_reshaped, y_test),
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )

    # Evaluate and plot results
    test_loss, test_mse = final_model.evaluate(X_test_reshaped, y_test, verbose=1)
    print(f"Test MSE: {test_mse}")

    predictions = final_model.predict(X_test_reshaped)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.title('True Values vs Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Close Change Percentage')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
