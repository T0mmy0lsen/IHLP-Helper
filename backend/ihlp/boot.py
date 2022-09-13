import time
import pickle
import re

from ihlp.model.svm import SVM


class Boot:

    labels_time = None
    labels_responsible = None

    path_pickle_time = 'C:/Git/ihlp-helper/pickle/time/4B0BE2C3'
    path_pickle_responsible = 'C:/Git/ihlp-helper/pickle/responsible/4B0BE2C3'

    # path_pickle_time = 'C:/Git/ihlp-helper/backend/predict/data/output/pickle/time/4B0BE2C3'
    # path_pickle_responsible = 'C:/Git/ihlp-helper/backend/predict/data/output/pickle/time/4B0BE2C3'
    # path_pickle_time = 'C:/Users/tool/git/ihlp-helper/pickle/time/4B0BE2C3'
    # path_pickle_responsible = 'C:/Users/tool/git/ihlp-helper/pickle/responsible/4B0BE2C3'

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
            self.svm_user = SVM().load(type='responsible')
            self.label_encoder_time = self.svm_time.le
            self.label_encoder_responsible = self.svm_user.le
            self.svm_loaded = True
        else:
            self.svm_loaded = False
            with open(self.path_pickle_time_le, 'rb') as pickle_file:
                self.label_encoder_time = pickle.load(pickle_file)
            with open(self.path_pickle_responsible_le, 'rb') as pickle_file:
                self.label_encoder_responsible = pickle.load(pickle_file)

        toc = time.perf_counter()
        print(f"Booted in {toc - tic:0.4f} seconds")

    def users(self):

        from ihlp.models import Machine

        # Check and create machines, s.t. we have no missing.
        for responsible in self.label_encoder_responsible.classes_:
            responsible = re.sub("[^a-zA-Z0-9]+", "", responsible).lower()
            results = Machine.objects.filter(user=responsible).first()
            if results is not None:
                continue
            else:
                Machine.objects.create(user=responsible)

    def reboot(self):

        from ihlp.models import Schedule, Machine, Slot

        # Remove all Slots relations
        [m.slots.clear() for m in Machine.objects.all()]
        Slot.objects.all().delete()

        [m.machines.clear() for m in Schedule.objects.all()]
        Schedule.objects.all().delete()
        Machine.objects.all().delete()

        schedules = [
            Schedule.objects.create(name='SDU IT'),
            Schedule.objects.create(name='SDU Digital'),
            Schedule.objects.create(name='Servicedesk'),
        ]

        machines = [
            'jpanduro',
            'njespersen',
            'kruchov',
            'mark',
            'tchristensen',
            'tvn',
            'aer',
            'afredsl',
            'agd',
        ]

        for i, machine in enumerate(machines):
            m = Machine.objects.create(user=machine)
            schedules[i % len(schedules)].machines.add(m)









