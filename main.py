# This is a fun Python script.

from model.preprocess import Preprocess


def preprocess():
    Preprocess({
        'beautify': True
    })


def run():
    preprocess()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()