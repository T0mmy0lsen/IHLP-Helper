from predict.model.model_keywords import ModelKeywords
from predict.model.model_trivial import ModelTrivial
from predict.model.model_cnn import ModelCNN
from predict.model.model_svm import ModelSVM
from predict.model.preprocess import Preprocess
from predict.model.prepare import Prepare
from predict.model.shared import SharedDict
from predict.model.wordembedding import WordEmbedding, WordEmbeddingLoader


def run(
        category_type='time',  # responsible or time
        do_run_trivial=False,
        do_run_keyword=False,
        do_run_svm=True,
        do_run_cnn=False
    ):

    shared = SharedDict().default()

    # The job of Preprocess is to process the text s.t. its ready for the model.
    Preprocess(shared)

    # The job of Prepare is to create the text and label columns.
    Prepare(
        shared,
        category_type=category_type,
        label_index=category_type
    ).fetch()

    if do_run_trivial:
        run_trivial(shared)
    if do_run_keyword:
        run_keywords(shared)
    if do_run_svm:
        run_svm(shared, category_type)
    if do_run_cnn:
        run_cnn(shared)


def run_svm(shared, category_type):
    ModelSVM(shared, category_type)


def run_keywords(shared):
    ModelKeywords(shared)


def run_trivial(shared):
    ModelTrivial(shared)


def run_cnn(shared):
    WordEmbedding(shared)
    WordEmbeddingLoader(shared)
    ModelCNN(shared)


if __name__ == '__main__':
    run()