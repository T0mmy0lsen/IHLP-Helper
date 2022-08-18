# This is a fun Python script.

import numpy as np
from nltk import FreqDist

from predict.model.model_cnn import ModelCNN
from predict.model.model_keywords import ModelKeywords
from predict.model.model_svm import ModelSVM

from predict.model.preprocess import Preprocess

# from model.train import Train
# from model.wordembedding import WordEmbedding, WordEmbeddingLoader

from predict.model.model_trivial import ModelTrivial
from predict.model.shared import SharedDict
from predict.model.prepare import Prepare
from predict.model.wordembedding import WordEmbedding, WordEmbeddingLoader


def run():

    category_type = 'responsible'

    shared = SharedDict().default()

    # The job of Preprocess is to process the text s.t. its ready for the model.
    Preprocess(shared)
    # The job of Prepare is to create the text and label columns.
    # If text or label are derived by some logic - it should be placed here.
    # Prepare(shared).fetch(amount=86000, categorical_index=False)

    Prepare(shared, category_type=category_type, label_index=category_type).fetch(
        amount=86000,
        categorical_index=False,
        lang=None,
        # boost=True,
        # roll=True,
        # top=4,
        # multiplier=4
    )

    # data_dict = get_data(shared)
    # get_stats(data_dict, shared)

    # run_trivial(shared)
    # run_keywords(shared)
    run_svm(shared, category_type)
    # run_cnn(shared)

    # ---------------------------------------------------------------------------------------------------------
    # Trivial       -   4 categories; with stemmer 0.25, without stemmer 0.26, new labels 0.24
    # Keyswords     -   4 categories; with stemmer 0.52, without stemmer 0.52, new labels 0.78
    # SVM           -   4 categories; with stemmer 0.69, without stemmer 0.69, new labels 0.88

    # ---------------------------------------------------------------------------------------------------------
    # Trivial       -   164 categories; 0.01
    # Keyswords     -   164 categories; 0.18
    # SVM           -   164 categories; 0.33


def run_svm(shared, category_type):
    ModelSVM(shared, category_type)


def run_keywords(shared):
    ModelKeywords(shared)


def run_trivial(shared):
    ModelTrivial(shared)


def run_cnn(shared):
    # This creates the word embedding.
    # The word embedding does not rely on Prepare. This means any Preprocess'ed data can be used to train the word embedder.
    WordEmbedding(shared)
    # Load in the created word embedder and create the Word Embedding layer.
    WordEmbeddingLoader(shared)
    # Train the model.
    ModelCNN(shared)
    pass


def get_data(shared):

    data_dict = {}
    for i in shared.categories:
        data_dict[i] = []
    for idx, el in enumerate(shared.x_train):
        data_dict[shared.y_train[idx]].append(el)
    for idx, el in enumerate(shared.x_validate):
        data_dict[shared.y_validate[idx]].append(el)
    return data_dict


def get_stats(data_dict, shared, num_words=1000):

    freq_dict = {}
    freq_dict_list = {}
    freq_dict_unique = {}

    for k in sorted(data_dict, key=lambda k: len(data_dict[k]), reverse=True):
        freq_dist = FreqDist(" ".join(data_dict[k]).split(" "))
        freq_dict[k] = freq_dist.most_common(num_words)
        freq_dict_list[k] = np.array([e[0] for e in freq_dict[k]])

    for k in freq_dict.keys():
        not_in = []
        for k_not_in in freq_dict_list.keys():
            if k != k_not_in:
                not_in = np.concatenate((not_in, freq_dict_list[k_not_in]))
        freq_dict_unique[k] = [e for i, e in enumerate(freq_dict[k]) if freq_dict_list[k][i] not in not_in]

    for idx, k in enumerate(freq_dict.keys()):
        print(len(data_dict[k]), "\t", shared.categories[idx], freq_dict_unique[k][:5])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()