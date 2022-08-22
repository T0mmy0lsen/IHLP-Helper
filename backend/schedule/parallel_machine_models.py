import numpy as np
import json
from django.forms import model_to_dict

from app import settings
from ihlp.models import Schedule, Machine, Slot, Predict


def run(text=None):

    # Fist we predict time and responsible

    # predict_responsible = [.12, .23, .01, .16, .11]
    # predict_responsible_top_k = np.argpartition(predict_responsible, -3)[-3:]
    # predict_time = np.argpartition([.1, .2, .4, .2, .1], -1)[-1:]

    if text is None:
        text = "Jeg kan ikke logge ind på Blackboard."

    predict_time_prob = settings.svm_time.predict(text)[0]
    predict_time = predict_time_prob[0]
    predict_time = int(predict_time)
    predict_responsible_prob = settings.svm_responsible.predict(text)
    predict_responsible = predict_responsible_prob[0]

    predict = Predict.objects.create(
        time=predict_time,
        responsible=predict_responsible
    )

    predict.save()

    # The user selects a responsible, which maps to a machine.
    user = "tool"

    machine = Machine.objects.filter(user=user).first()
    schedule = machine.schedule_set.first()

    print(machine)
    print(schedule)

    machines = schedule.machines.all()
    
    # Get all existing slots from the Schedule and add the new slot.
    s = Slot.objects.create(user=user, time=predict_time, index=0)

    slots = np.concatenate([list(m.slots.all()) for m in machines])
    slots = np.concatenate((slots, [s]))

    response = {
        "predict_time_prob": list(predict_time_prob),
        "predict_responsible_prob": list(predict_responsible_prob),
        "schedule": list(longest_processing_time_first(machines, slots))
    }

    return json.dumps(response)

def longest_processing_time_first(m_machines=None, n_slots=None):

    m_machines_arrays = np.array([[0] * len(n_slots) for _ in range(len(m_machines))])
    n_slots_sorted = sorted(n_slots, key=lambda x: x.time, reverse=True)

    # Remove all Slots relations
    [m.slots.clear() for m in m_machines]

    for i, n_job in enumerate(n_slots_sorted):

        index_with_lowest_sum = int(np.argmin(np.array(m_machines_arrays).sum(axis=1)))

        m_machines_arrays[index_with_lowest_sum][i] = n_job.time
        m_machines[index_with_lowest_sum].slots.add(n_job)

        n_job.index = i
        n_job.user = m_machines[index_with_lowest_sum].user
        n_job.save()

    return m_machines_arrays