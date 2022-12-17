import numpy as np
import pandas as pd
import tensorflow as tf


class ModelCNN_TDIDF:

    def __init__(self):

        USE_U_NET = True
        LEARNING_RATE = 0.0005

        DATA_PATH_X_TRAIN = 'data/0_baselines/data_x_train.csv'
        DATA_PATH_Y_TRAIN = 'data/0_baselines/data_y_train.csv'
        DATA_PATH_X_TEST = 'data/0_baselines/data_x_test.csv'
        DATA_PATH_Y_TEST = 'data/0_baselines/data_y_test.csv'

        x_train = pd.read_csv(DATA_PATH_X_TRAIN).to_numpy()
        y_train = pd.read_csv(DATA_PATH_Y_TRAIN).to_numpy()
        x_test = pd.read_csv(DATA_PATH_X_TEST).to_numpy()
        y_test = pd.read_csv(DATA_PATH_Y_TEST).to_numpy()

        def build_model(layer_first, start_neurons=8, kernel_size=4, dropout=0.25):

            conv1 = tf.keras.layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(layer_first)
            conv1 = tf.keras.layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(conv1)
            pool1 = tf.keras.layers.MaxPooling1D(2)(conv1)
            pool1 = tf.keras.layers.Dropout(0.25)(pool1)

            conv2 = tf.keras.layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(pool1)
            conv2 = tf.keras.layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(conv2)
            pool2 = tf.keras.layers.MaxPooling1D(2)(conv2)
            pool2 = tf.keras.layers.Dropout(dropout)(pool2)

            conv3 = tf.keras.layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(pool2)
            conv3 = tf.keras.layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(conv3)
            pool3 = tf.keras.layers.MaxPooling1D(2)(conv3)
            pool3 = tf.keras.layers.Dropout(dropout)(pool3)

            conv4 = tf.keras.layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(pool3)
            conv4 = tf.keras.layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(conv4)
            pool4 = tf.keras.layers.MaxPooling1D(2)(conv4)
            pool4 = tf.keras.layers.Dropout(dropout)(pool4)

            # Middle
            convm = tf.keras.layers.Conv1D(start_neurons * 16, kernel_size, activation="relu", padding="same")(pool4)
            convm = tf.keras.layers.Conv1D(start_neurons * 16, kernel_size, activation="relu", padding="same")(convm)

            deconv4 = tf.keras.layers.Conv1DTranspose(start_neurons * 8, kernel_size, strides=2, padding="same")(convm)
            uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
            uconv4 = tf.keras.layers.Dropout(dropout)(uconv4)
            uconv4 = tf.keras.layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(uconv4)
            uconv4 = tf.keras.layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(uconv4)

            deconv3 = tf.keras.layers.Conv1DTranspose(start_neurons * 4, kernel_size, strides=2, padding="same")(uconv4)
            uconv3 = tf.keras.layers.concatenate([deconv3, conv3])
            uconv3 = tf.keras.layers.Dropout(dropout)(uconv3)
            uconv3 = tf.keras.layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(uconv3)
            uconv3 = tf.keras.layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(uconv3)

            deconv2 = tf.keras.layers.Conv1DTranspose(start_neurons * 2, kernel_size, strides=2, padding="same")(uconv3)
            uconv2 = tf.keras.layers.concatenate([deconv2, conv2])
            uconv2 = tf.keras.layers.Dropout(dropout)(uconv2)
            uconv2 = tf.keras.layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(uconv2)
            uconv2 = tf.keras.layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(uconv2)

            deconv1 = tf.keras.layers.Conv1DTranspose(start_neurons * 1, kernel_size, strides=2, padding="same")(uconv2)
            uconv1 = tf.keras.layers.concatenate([deconv1, conv1])
            uconv1 = tf.keras.layers.Dropout(dropout)(uconv1)
            uconv1 = tf.keras.layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(uconv1)
            uconv1 = tf.keras.layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(uconv1)

            output_layer = tf.keras.layers.GlobalMaxPool1D()(uconv1)

            # output_layer = tf.keras.layers.Dense(128, activation='relu')(output_layer)
            # output_layer = tf.keras.layers.Conv1D(1, 1, padding="same", activation="sigmoid")(uconv1)

            return output_layer

        if USE_U_NET:
            x_train = np.array([e[:3376] for e in x_train])
            x_test = np.array([e[:3376] for e in x_test])
            layer_first = tf.keras.Input(shape=(3376, 1), dtype="float64")
            output_layer = build_model(layer_first)
        else:
            layer_first = tf.keras.Input(shape=(len(x_train[0]), 1), dtype="float64")
            x = tf.keras.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='relu')(layer_first)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.Conv1D(filters=16, kernel_size=4, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv1D(filters=16, kernel_size=4, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')(x)
            x = tf.keras.layers.GlobalMaxPool1D()(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            output_layer = tf.keras.layers.Dense(128, activation='relu')(x)


        preds = tf.keras.layers.Dense(len(np.unique(np.concatenate((y_train, y_test)))), activation="softmax")(output_layer)
        model = tf.keras.Model(layer_first, preds)

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=['accuracy']
        )

        model.fit(x_train, y_train, batch_size=16, epochs=50, validation_data=(x_test, y_test))

ModelCNN_TDIDF()