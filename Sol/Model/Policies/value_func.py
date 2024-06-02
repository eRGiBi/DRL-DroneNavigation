import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt


def filter_lines_by_length(file_path, length):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip().split(',') for line in lines if len(line.strip().split(',')) == length]


def read_data():
    rollouts = pd.DataFrame()

    for file in os.listdir("./Sol/rollouts"):
        print(file)
        if file.endswith(".csv"):
            rollouts = pd.concat([rollouts, pd.read_csv(f"Sol/rollouts/{file}")], ignore_index=True)
        elif file.endswith(".gz"):
            rollouts = pd.concat([rollouts, pd.read_csv(f"Sol/rollouts/{file}", compression='gzip')], ignore_index=True)
        elif file.endswith(".txt"):
            filtered_lines = filter_lines_by_length(f"Sol/rollouts/{file}", 13)
            if filtered_lines:
                filtered_df = pd.DataFrame(filtered_lines)
                rollouts = pd.concat([rollouts, filtered_df], ignore_index=True)

    print("Rollouts DataFrame after processing all files:")
    print(rollouts.head())

    rollouts.replace([np.inf, -np.inf], np.nan, inplace=True)
    rollouts.dropna(inplace=True)

    rollouts = rollouts.dropna()
    rollouts = rollouts.to_numpy(dtype=float)

    x = rollouts[:, :-1]
    y = rollouts[:, -1]

    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"x[0]: {x[0]}, y[0]: {y[0]}")
    print(f"type(x[0]): {type(x[0])}, type(y[0]): {type(y[0])}")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


def create_model(input_shape):
    initializer = tf.keras.initializers.orthogonal()
    model = tf.keras.models.Sequential([
        layers.Dense(512, activation=tf.keras.activations.tanh, input_shape=(input_shape,), kernel_initializer=initializer),
        layers.Dense(512, activation=tf.keras.activations.tanh, kernel_initializer=initializer),
        layers.Dense(256, activation=tf.keras.activations.tanh, kernel_initializer=initializer),
        layers.Dense(1, activation='linear', kernel_initializer=initializer)
    ])
    return model


def main():
    x_train, x_test, y_train, y_test = read_data()
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print(f"x_train[0]: {x_train[0]}, y_train[0]: {y_train[0]}")

    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():

    model = create_model(12)

    model.compile(optimizer=tf.keras.optimizers.Adam(2.5e-4),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'],
                  )

    callbacks = [
        # keras.callbacks.ModelCheckpoint(filepath="checkpoint_dir/saved-model.hdf5",
        #                                 save_freq='epoch', monitor='loss', mode='min',
        #                                 save_best_only=True, verbose=0),
        # keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=2, verbose=1)
    ]

    start = time.perf_counter()
    # with strategy.scope():

    history = model.fit(np.array(x_train), np.array(y_train),
                        validation_data=(np.array(x_test), np.array(y_test)),

                        epochs=30, batch_size=64, verbose=2, callbacks=callbacks)

    print(f"Training time: {time.perf_counter() - start} seconds")

    test_loss, test_mse = model.evaluate(np.array(x_test), np.array(y_test), verbose=2)
    print(f'\nTest Mean Squared Error: {test_mse}')
    print(f'\nTest Loss: {test_loss}')

    predictions = model.predict(np.array(x_test))

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mean_squared_error'], label='Training MSE')
    plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
    plt.title('Training and Validation Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

