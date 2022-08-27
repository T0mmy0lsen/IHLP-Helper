import numpy as np
import json
import pickle
from django.forms import model_to_dict

from sklearn import preprocessing
from app.settings import BOOT
from ihlp.model.svm import SVM
from ihlp.models import Schedule, Machine, Slot, Predict
from ihlp.models_ihlp import Request


def setPredict(id=None, user=None, time=None):

    machine = Machine.objects.filter(user=user).first()
    schedule = machine.schedule_set.first()
    machines = schedule.machines.all()

    # Get all existing slots from the Schedule and add the new slot.
    s = Slot.objects.create(user=user, time=time, index=0)

    slots = np.concatenate([list(m.slots.all()) for m in machines])
    slots = np.concatenate((slots, [s]))

    longest_processing_time_first(machines, slots, save=True)


def getPredict(id=None):

    # Fist we predict time and responsible

    if id is None:
        id = 3710881

    request = Request.objects.filter(id=id).using('ihlp').first()
    text = request.description

    if BOOT is not None and BOOT.svm_loaded:
        # To sync Machines with the model, run boot.create_responsible()
        predict_time_prob = BOOT.svm_time.predict(text)
        predict_responsible_prob = BOOT.svm_responsible.predict(text)
    elif BOOT is not None:
        # For debug purposes
        predict_time_prob = [.1, .2, .4, .2, .1]
        predict_responsible_prob = np.random.random(size=9)
    else:
        return None

    predict_time_indexes = np.argpartition(predict_time_prob, -3)[-3:]
    predict_time_indexes = [int(t) for t in predict_time_indexes]
    predict_responsible_indexes = np.argpartition(predict_responsible_prob, -3)[-3:]
    predict_responsible_indexes = [int(t) for t in predict_responsible_indexes]

    if BOOT is not None and not BOOT.svm_loaded:
        # For debug purposes
        BOOT.label_encoder_time = preprocessing.LabelEncoder()
        BOOT.label_encoder_time.fit([1, 2, 3, 4, 5])
        BOOT.label_encoder_responsible = preprocessing.LabelEncoder()
        BOOT.label_encoder_responsible.fit([
            'jpanduro',
            'njespersen',
            'kruchov',
            'mark',
            'tchristensen',
            'tvn',
            'aer',
            'afredsl',
            'agd',
        ])

    labels_time = BOOT.label_encoder_time.inverse_transform(predict_time_indexes)
    labels_responsible = BOOT.label_encoder_responsible.inverse_transform(predict_responsible_indexes)

    # The user selects a responsible, which maps to a machine.
    responses = []
    for responsible in labels_responsible:

        # predict = Predict.objects.create(time=predict_time, responsible=predict_responsible)
        # predict.save()

        user = responsible.lower()
        machine = Machine.objects.filter(user=user).first()

        if machine is None:
            responses.append({
                "error": f"User \'{user}\' not found in Machines"
            })
            continue

        schedule = machine.schedule_set.first()

        machines = schedule.machines.all()
    
        # Get all existing slots from the Schedule and add the new slot.
        s = Slot.objects.create(user=user, time=labels_time[0], index=id)

        slots = np.concatenate([list(m.slots.all()) for m in machines])
        slots = np.concatenate((slots, [s]))

        schedules = list(longest_processing_time_first(machines, slots, id=id))
        team_users = [m.user for m in machines if m.user in BOOT.label_encoder_responsible.classes_]
        team_indexes = BOOT.label_encoder_responsible.transform(team_users)
        responsible_index = BOOT.label_encoder_responsible.transform([responsible])[0]

        result = {
            "sum": sum(sum([int(_m['time']) for _m in m]) for m in schedules) - int(labels_time[0]),
            "user": user,
            "time": int(labels_time[0]),
            "user_prob": round(predict_responsible_prob[responsible_index], 3),
            "team": schedule.name,
            "team_prob": round(sum(predict_responsible_prob[team_indexes]), 3),
            "schedule": {}
        }

        for i, machine in enumerate(machines):
            result['schedule'][machine.user] = [m for m in schedules[i]]

        responses.append(result)

    return json.dumps(responses)


def longest_processing_time_first(m_machines=None, n_slots=None, save=False, id=None):

    m_machines_arrays = np.array([[{'time': 0}] * len(n_slots) for _ in range(len(m_machines))])
    n_slots_sorted = sorted(n_slots, key=lambda x: x.time, reverse=True)

    if save:
        # Remove all Slots relations
        [m.slots.clear() for m in m_machines]

    for i, n_slot in enumerate(n_slots_sorted):

        index_with_lowest_sum = int(np.argmin(np.array([[_m['time'] for _m in m] for m in m_machines_arrays]).sum(axis=1)))
        m_machines_arrays[index_with_lowest_sum][i] = {
            'time': int(n_slot.time),
            'index': n_slot.index,
            'current': n_slot.index == id
        }

        if save:
            m_machines[index_with_lowest_sum].slots.add(n_slot)
            n_slot.index = i
            n_slot.user = m_machines[index_with_lowest_sum].user
            n_slot.save()

    return [m for m in m_machines_arrays]