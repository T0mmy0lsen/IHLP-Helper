import time
import pickle

from ihlp.model.svm import SVM


class Boot:

    labels_time = None
    labels_responsible = None

    # path_pickle_time = 'C:/Git/ihlp-helper/backend/predict/data/output/pickle/time/4B0BE2C3'
    path_pickle_time = 'C:/Git/ihlp-helper/pickle/time/4B0BE2C3'
    # path_pickle_responsible = 'C:/Git/ihlp-helper/backend/predict/data/output/pickle/time/4B0BE2C3'
    path_pickle_responsible = 'C:/Git/ihlp-helper/pickle/responsible/4B0BE2C3'

    path_pickle_time_clf = f"{path_pickle_time}/tfidf.pickle"
    path_pickle_time_tfi = f"{path_pickle_time}/clf.pickle"
    path_pickle_time_le = f"{path_pickle_time}/le.pickle"

    path_pickle_responsible_clf = f"{path_pickle_responsible}/tfidf.pickle"
    path_pickle_responsible_tfi = f"{path_pickle_responsible}/clf.pickle"
    path_pickle_responsible_le = f"{path_pickle_responsible}/le.pickle"

    def __init__(self, debug=True):
        tic = time.perf_counter()

        # Load SVM
        if not debug:
            self.svm_time = SVM().load(type='time')
            self.svm_responsible = SVM().load(type='responsible')
            self.svm_loaded = True
        else:
            self.svm_loaded = False
            with open(self.path_pickle_time_le, 'rb') as pickle_file:
                self.label_encoder_time = pickle.load(pickle_file)
            with open(self.path_pickle_responsible_le, 'rb') as pickle_file:
                self.label_encoder_responsible = pickle.load(pickle_file)

        toc = time.perf_counter()
        print(f"Booted in {toc - tic:0.4f} seconds")

    def create_responsible(self):
        from ihlp.models import Machine

        # Check and create machines, s.t. we have no missing.
        for responsible in self.label_encoder_responsible.classes_:
            responsible = responsible.lower()
            results = Machine.objects.filter(user=responsible).first()
            if results is not None:
                continue
            else:
                Machine.objects.create(user=responsible)




