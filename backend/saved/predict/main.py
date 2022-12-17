from predict.model.model_trivial import ModelTrivial
from predict.model.model_cnn import ModelCNN
from predict.model.preprocess import Preprocess
from predict.model.prepare import Prepare
from predict.model.shared import SharedDict
from predict.model.wordembedding import WordEmbedding, WordEmbeddingLoader


def run(
        category_type='responsible',    # Choose responsible or time
        do_run_trivial=False,
        do_run_cnn=True,
    ):

    shared = SharedDict().revised()

    # The job of Preprocess is to process the text s.t. its ready for the model.
    Preprocess(shared)

    # The job of Prepare is to create the text and label columns.
    Prepare(
        shared,
        category_type=category_type
    ).fetch(
        top=100,
        categorical=True,
        categorical_index=True,
        lang='da',
    )

    if do_run_trivial:
        run_trivial(shared, category_type)
    if do_run_cnn:
        run_cnn(shared)


def run_trivial(shared, category_type):
    ModelTrivial(shared, category_type)


def run_cnn(shared):
    WordEmbedding(shared)
    WordEmbeddingLoader(shared)
    ModelCNN(shared)


if __name__ == '__main__':
    run()