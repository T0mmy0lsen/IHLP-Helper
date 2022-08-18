
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


class ModelCNN:

    def __init__(self, shared):
        if shared.model == 1:
            self.model_01(shared)


    def model_01(self, shared):

        def bulk_vectorize(data, size=5000):
            output = None
            prev_i = 0
            for i, s in enumerate(tqdm(data)):
                if i > 0 and i % size == 0:
                    # part_output = shared.vectorizer(np.array([[s] for s in data[prev_i:i]])).numpy()
                    part_output = shared.vectorizer(data[prev_i:i])
                    if output is None:
                        output = part_output[:]
                    else:
                        output = np.vstack((output, part_output))
                    prev_i = i
            # part_output = shared.vectorizer(np.array([[s] for s in data[prev_i:]])).numpy()
            part_output = shared.vectorizer(data[prev_i:])
            if output is None:
                output = part_output[:]
            else:
                output = np.vstack((output, part_output))
            return output

        # Use wordembedding
        # """
        layer_first = tf.keras.Input(shape=(None,), dtype="int64")
        layer_model_build = shared.layer(layer_first)
        # """

        # Use tfidf
        #
        """"
        tfidf_vectorizer = TfidfVectorizer(max_df=.9, max_features=1536, min_df=0.01, ngram_range=(1, 5))
        tfidf_vectorizer.fit_transform(shared.x_train)
        tf_len = len(tfidf_vectorizer.vocabulary_)
        print(tf_len)
        
        layer_first = tf.keras.Input(shape=(tf_len, 1), dtype="float64")
        layer_model_build = layer_first
        
        def vectorizer(data):
            return tfidf_vectorizer.transform(data).todense()

        shared.vectorizer = vectorizer
        """

        """
        lx = layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='relu')(embedded_sequences)
        lx = layers.MaxPooling1D(pool_size=2)(lx)
        lx = layers.Conv1D(filters=16, kernel_size=4, padding='same', activation='relu')(lx)
        lx = layers.Conv1D(filters=16, kernel_size=4, padding='same', activation='relu')(lx)
        
        lx = layers.Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')(lx)
        lx = layers.GlobalMaxPool1D()(lx)
        lx = layers.Dense(256, activation='relu')(lx)
        lx = layers.Dense(128, activation='relu')(lx)
        """

        def build_model(input_layer, start_neurons=8, kernel_size=4, dropout=0.25):
            conv1 = layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(input_layer)
            conv1 = layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(conv1)
            pool1 = layers.MaxPooling1D(2)(conv1)
            pool1 = layers.Dropout(0.25)(pool1)

            conv2 = layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(pool1)
            conv2 = layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(conv2)
            pool2 = layers.MaxPooling1D(2)(conv2)
            pool2 = layers.Dropout(dropout)(pool2)

            conv3 = layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(pool2)
            conv3 = layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(conv3)
            pool3 = layers.MaxPooling1D(2)(conv3)
            pool3 = layers.Dropout(dropout)(pool3)

            conv4 = layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(pool3)
            conv4 = layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(conv4)
            pool4 = layers.MaxPooling1D(2)(conv4)
            pool4 = layers.Dropout(dropout)(pool4)

            # Middle
            convm = layers.Conv1D(start_neurons * 16, kernel_size, activation="relu", padding="same")(pool4)
            convm = layers.Conv1D(start_neurons * 16, kernel_size, activation="relu", padding="same")(convm)

            deconv4 = layers.Conv1DTranspose(start_neurons * 8, kernel_size, strides=2, padding="same")(convm)
            uconv4 = layers.concatenate([deconv4, conv4])
            uconv4 = layers.Dropout(dropout)(uconv4)
            uconv4 = layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(uconv4)
            uconv4 = layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(uconv4)

            deconv3 = layers.Conv1DTranspose(start_neurons * 4, kernel_size, strides=2, padding="same")(uconv4)
            uconv3 = layers.concatenate([deconv3, conv3])
            uconv3 = layers.Dropout(dropout)(uconv3)
            uconv3 = layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(uconv3)
            uconv3 = layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(uconv3)

            deconv2 = layers.Conv1DTranspose(start_neurons * 2, kernel_size, strides=2, padding="same")(uconv3)
            uconv2 = layers.concatenate([deconv2, conv2])
            uconv2 = layers.Dropout(dropout)(uconv2)
            uconv2 = layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(uconv2)
            uconv2 = layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(uconv2)

            deconv1 = layers.Conv1DTranspose(start_neurons * 1, kernel_size, strides=2, padding="same")(uconv2)
            uconv1 = layers.concatenate([deconv1, conv1])
            uconv1 = layers.Dropout(dropout)(uconv1)
            uconv1 = layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(uconv1)
            uconv1 = layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(uconv1)

            output_layer = layers.GlobalMaxPool1D()(uconv1)

            # output_layer = layers.Dense(128, activation='relu')(output_layer)
            # output_layer = layers.Conv1D(1, 1, padding="same", activation="sigmoid")(uconv1)

            return output_layer

        output_layer = build_model(layer_model_build)

        preds = layers.Dense(len(shared.categories), activation="softmax")(output_layer)
        model = tf.keras.Model(layer_first, preds)
        # model.summary()

        x_train = bulk_vectorize(shared.x_train)
        x_val = bulk_vectorize(shared.x_validate)

        y_train = np.array(shared.y_train)
        y_val = np.array(shared.y_validate)

        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="Adamax", metrics=["acc"]
        )

        print(np.unique(y_train, return_counts=True))
        print(np.unique(y_val, return_counts=True))

        model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_val, y_val))
        model.save()

        string_input = tf.keras.Input(shape=(1,), dtype="string")
        v = shared.vectorizer(string_input)
        preds = model(v)
        end_to_end_model = tf.keras.Model(string_input, preds)

        probabilities = end_to_end_model.predict(
            [["Jeg skal bruge en ny computer"]]
        )

        print(shared.categories[np.argmax(probabilities[0])])