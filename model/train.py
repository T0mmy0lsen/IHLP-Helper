
import numpy as np
import tensorflow as tf
from keras import layers


class Train:

    def __init__(self, shared, x_train, y_train, x_validate, y_validate, categories):
        if shared.model == 1:
            self.model_01(shared, x_train, y_train, x_validate, y_validate, categories)


    def model_01(self, shared, x_train, y_train, x_validate, y_validate, categories):

        int_sequences_input = tf.keras.Input(shape=(None,), dtype="int64")
        embedded_sequences = shared.layer(int_sequences_input)
        lx = layers.Conv1D(256, 5, activation="relu")(embedded_sequences)
        lx = layers.GlobalAveragePooling1D()(lx)
        lx = layers.Dense(128, activation="relu")(lx)

        preds = layers.Dense(len(categories), activation="softmax")(lx)
        model = tf.keras.Model(int_sequences_input, preds)
        # model.summary()

        x_train = shared.vectorizer(np.array([[s] for s in x_train])).numpy()
        x_val = shared.vectorizer(np.array([[s] for s in x_validate])).numpy()

        y_train = np.array(y_train)
        y_val = np.array(y_validate)

        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
        )

        model.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_val, y_val))

        string_input = tf.keras.Input(shape=(1,), dtype="string")
        v = shared.vectorizer(string_input)
        preds = model(v)
        end_to_end_model = tf.keras.Model(string_input, preds)

        probabilities = end_to_end_model.predict(
            [["Jeg skal bruge en ny computer"]]
        )

        print(categories[np.argmax(probabilities[0])])