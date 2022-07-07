"""

This creates the Word Embedding.
The config.py refers to the input and output destination of this file.
The transform.py creates the input-file by transforming the raw data from the database.

"""

import io
import os
import re
import shutil

import tqdm

import config
import string

import numpy as np
import pandas as pd
import tensorflow as tf


class Word2Vec(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        num_ns = 4
        self.target_embedding = tf.keras.layers.Embedding(vocab_size,
                                                          embedding_dim,
                                                          input_length=1,
                                                          name="w2v_embedding")
        self.context_embedding = tf.keras.layers.Embedding(vocab_size,
                                                           embedding_dim,
                                                           input_length=num_ns + 1)

    def call(self, pair):
        target, context = pair
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        return dots


class WordEmbeddingLoader:

    def __init__(self, shared=None, dim=128, x_train=None):

        vectorizer = tf.keras.layers.TextVectorization(standardize=None, max_tokens=200000, output_sequence_length=100)
        text_ds = tf.data.Dataset.from_tensor_slices(x_train).batch(128)
        vectorizer.adapt(text_ds)

        voc = vectorizer.get_vocabulary()
        word_index = dict(zip(voc, range(len(voc))))

        embeddings_index = {}

        skipped = 0
        we_path = f'{config.BASE_PATH}/model/output/wordembedding/IHLP_{shared.hashed}_{dim}.txt'
        with open(we_path) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                if len(coefs) > 0:
                    embeddings_index[word] = coefs
                else:
                    # For some reason we got a two-word word in the Word Embedding.
                    skipped += 1

        print("Found %s word vectors." % len(embeddings_index))
        print("Skipped %s word vectors." % skipped)

        num_tokens = len(voc) + 2
        hits = 0
        misses = 0
        missed = []

        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
                missed.append(word)

        print("Converted %d words (%d misses)" % (hits, misses))
        print(missed[:10])

        embedding_layer = tf.keras.layers.Embedding(
            num_tokens,
            dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )

        shared.set_word_embedding_layer(embedding_layer)
        shared.set_vectorizer(vectorizer)


class WordEmbedding:

    SEED = 1337
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, shared=None):

        self.shared = shared

        path = f'{config.BASE_PATH}/model/output/wordembedding/IHLP_{self.shared.hashed}_{self.shared.word_embedding_dim}.txt'
        if os.path.exists(path):
            msg = f'Can not override existing word embedding {path}'
            print(msg)
            return

        file_path = f'{config.BASE_PATH}/model/tmp/wordembedding/{self.shared.hashed}.csv'

        if not os.path.exists(file_path):
            tmp_line = []
            for idx_name, name in enumerate(self.shared.dfs_names):
                for df_index in self.shared.dfs_index[idx_name]:
                    df_tmp = pd.read_csv(f'{config.BASE_PATH}/model/output/preprocessed/{self.shared.hashed}/{name}_{df_index}.csv')
                    df_tmp = df_tmp.fillna('')
                    for idx_row, row in df_tmp.iterrows():
                        tmp_line.append(row[df_index])
            tmp = pd.DataFrame(tmp_line, columns=['text'])
            tmp.to_csv(file_path, header=False, index=False, encoding='utf-8')
        else:
            print(f'There is already a tmp-file for the Word Embedding')
            print(f'Remove the folder {self.shared.hashed}.csv file in /tmp/wordembedding to rebuild it')

        text_ds = tf.data.TextLineDataset(file_path).filter(lambda x: tf.cast(tf.strings.length(x), bool))

        # Define the vocabulary size and the number of words in a sequence.
        vocab_size = 200000
        sequence_length = 100

        # Use the `TextVectorization` layer to normalize, split, and map strings to
        # integers. Set the `output_sequence_length` length to pad all samples to the
        # same length.
        vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=None,
            max_tokens=vocab_size + 1,
            output_mode='int',
            output_sequence_length=sequence_length)

        vectorize_layer.adapt(text_ds.batch(1024))

        # Vectorize the data in text_ds.
        text_vector_ds = text_ds.batch(1024).prefetch(WordEmbedding.AUTOTUNE).map(vectorize_layer).unbatch()

        sequences = list(text_vector_ds.as_numpy_iterator())  # It gets slow from this point

        generate_path = f'{config.BASE_PATH}/model/tmp/generated/{self.shared.hashed}'
        if os.path.isdir(generate_path):
            shutil.rmtree(generate_path)
        os.makedirs(generate_path)

        targets, contexts, labels = self.generate_training_data(
            sequences=sequences,
            window_size=4,
            num_ns=8,
            vocab_size=vocab_size,
            seed=WordEmbedding.SEED)

        targets = np.array(targets)
        contexts = np.array(contexts)[:, :, 0]
        labels = np.array(labels)

        print('\n')
        print(f"targets.shape: {targets.shape}")
        print(f"contexts.shape: {contexts.shape}")
        print(f"labels.shape: {labels.shape}")

        BATCH_SIZE = 1024
        BUFFER_SIZE = 10000
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        word2vec = Word2Vec(vocab_size, self.shared.word_embedding_dim)
        word2vec.compile(optimizer='adam',
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        word2vec.fit(dataset, epochs=self.shared.word_embedding_epochs, callbacks=[tensorboard_callback])

        weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
        vocab = vectorize_layer.get_vocabulary()

        out_v = io.open(path, 'w', encoding='utf-8')

        for index, word in enumerate(vocab):
            if index == 0:
                continue
            vec = weights[index]
            out_v.write(f'{word} ' + ' '.join([str(x) for x in vec]) + "\n")
        out_v.close()

    def write_out(self, targets, contexts, labels, idx):

        # Targets
        str_data = "\n".join([str(x) for x in targets])
        if len(str_data) > 0:
            f = open(f'{config.BASE_PATH}/model/tmp/generated/{self.shared.hashed}/targets_{idx}.dat', 'a')
            f.write(str_data)
            f.close()

        # Context
        tmp = [e.numpy().flatten() for e in contexts]
        str_data = "\n".join([", ".join([str(y) for y in x]) for x in tmp])
        if len(str_data) > 0:
            f = open(f'{config.BASE_PATH}/model/tmp/generated/{self.shared.hashed}/contexts_{idx}.dat', 'a')
            f.write(str_data)
            f.close()

        # Labels
        tmp = [e.numpy().flatten() for e in labels]
        str_data = "\n".join([", ".join([str(y) for y in x]) for x in tmp])
        if len(str_data) > 0:
            f = open(f'{config.BASE_PATH}/model/tmp/generated/{self.shared.hashed}/labels_{idx}.dat', 'a')
            f.write(str_data)
            f.close()

    # Generates skip-gram pairs with negative sampling for a list of sequences
    # (int-encoded sentences) based on window size, number of negative samples
    # and vocabulary size.

    def generate_training_data(self, sequences, window_size, num_ns, vocab_size, seed):
        # Elements of each training example are appended to these lists.
        targets, contexts, labels = [], [], []
        idx = 0

        # Build the sampling table for `vocab_size` tokens.
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

        # Iterate over all sequences (sentences) in the dataset.
        for sequence in tqdm.tqdm(sequences):

            # Generate positive skip-gram pairs for a sequence (sentence).
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=vocab_size,
                sampling_table=sampling_table,
                window_size=window_size,
                negative_samples=0)

            # Iterate over each positive skip-gram pair to produce training examples
            # with a positive context word and negative samples.
            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(
                    tf.constant([context_word], dtype="int64"), 1)
                negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=vocab_size,
                    seed=WordEmbedding.SEED,
                    name="negative_sampling")

                # Build context and label vectors (for one target word)
                negative_sampling_candidates = tf.expand_dims(
                    negative_sampling_candidates, 1)

                context = tf.concat([context_class, negative_sampling_candidates], 0)
                label = tf.constant([1] + [0] * num_ns, dtype="int64")

                # Append each element from the training example to global lists.
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

            if idx % 100000 == 0:
                self.write_out(targets, contexts, labels, idx)
                targets, contexts, labels = [], [], []
            idx = idx + 1

        self.write_out(targets, contexts, labels, idx)
        return targets, contexts, labels
