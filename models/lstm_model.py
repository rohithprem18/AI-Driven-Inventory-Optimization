import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm(window: int) -> Sequential:
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(window, 1)),
            Dropout(0.2),
            LSTM(32),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def train_lstm(model: Sequential, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch: int) -> Sequential:
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch, verbose=0)
    return model


def forecast_lstm(model: Sequential, scaled: np.ndarray, window: int, steps: int) -> np.ndarray:
    input_seq = scaled[-window:].reshape(1, window, 1)
    preds = []
    for _ in range(steps):
        pred = model.predict(input_seq, verbose=0)[0, 0]
        preds.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
    return np.array(preds)
