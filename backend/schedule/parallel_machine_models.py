import numpy as np
import json

from sklearn import preprocessing
from app.settings import BOOT

from ihlp.models import Schedule, Machine, Slot
from ihlp.models_ihlp import Request


def setPredict(params=None):

    if not params['team_id']:
        machine = Machine.objects.filter(user=params['user']).first()
        schedule = machine.schedule_set.first()
    else:
        schedule = Schedule.objects.filter(id=params['team_id']).first()

    machines = schedule.machines.all()

    # Get all existing slots from the Schedule and add the new slot.
    if params['user'] is False:
        params['user'] = '-'
    s = Slot.objects.create(user=params['user'], time=params['time'], index=params['id'], keep=params['keep'])

    slots = np.concatenate([list(m.slots.all()) for m in machines])
    slots = np.concatenate((slots, [s]))

    longest_processing_time_first(machines, slots, save=True)


def getPredict(id=None):

    # Fist we predict time and user

    if id is None:
        id = 3710881

    request = Request.objects.filter(id=id).using('ihlp').first()
    text = request.description

    if BOOT is not None and BOOT.svm_loaded:
        # To sync Machines with the model, run boot.users()
        predict_time_prob = BOOT.svm_time.predict(text)[0]
        predict_user_prob = BOOT.svm_user.predict(text)[0]
    elif BOOT is not None:
        # For debug purposes
        # The labels for both time and users dictates what time and users we can use for the Slots
        # We may predict users that are not eligible, i.e. users that do not belong to a team.
        # We find teams with users that does not a probability score (because they are not in the model)
        predict_time_prob = [.1, .2, .4, .2, .1]
        predict_user_prob = np.random.random(size=9)
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
    else:
        return None


    # We find the top 3 users and time that has been predicted.
    # We get the indexes, s.t. we can use the label encoder to find the actual time and usernames

    predict_time_indexes = np.argpartition(predict_time_prob, -3)[-3:]
    predict_time_indexes = [int(t) for t in predict_time_indexes]

    predict_user_indexes = np.argpartition(predict_user_prob, -3)[-3:]
    predict_user_indexes = [int(t) for t in predict_user_indexes]

    labels_time = BOOT.label_encoder_time.inverse_transform(predict_time_indexes)
    labels_user = BOOT.label_encoder_responsible.inverse_transform(predict_user_indexes)

    # We find all users, aka. Machines, in our database
    machines = Machine.objects.filter(user__in=labels_user).all()

    responses = []
    for machine in machines:

        # Find the team, aka. Schedule, of the user and the rest of the users of that team.
        team_schedule = machine.schedule_set.first()
        if team_schedule is None:
            print(f'User: {machine.user} not in a team.')
            continue

        team_machines = team_schedule.machines.all()
    
        # Get all existing Slots from the Schedule and add the new slot.
        # We set the time to 1 for debugging purposes, else it should be = labels_time[0]
        s = Slot(user=None, time=labels_time[0], index=id, keep=False)

        slots = np.concatenate([list(m.slots.all()) for m in team_machines])
        slots = np.concatenate((slots, [s]))

        # Schedule all the slots using the new slot.
        schedules = list(longest_processing_time_first(team_machines, slots, id=id))

        result = dict({
            "time": int(s.time),
            "team": team_schedule.name,
            "team_id": team_schedule.id,
            "team_time": 0,
            "team_prob": 0,
            "schedule": {}
        })

        # We get the indexes for the users, s.t. we can find their probability.
        # We need this to evaluate the whole team.

        users = [m.user for m in team_machines]
        users_index = BOOT.label_encoder_responsible.transform(users)
        users_probs = predict_user_prob[users_index]

        for i, team_machine in enumerate(team_machines):
            result['schedule'][team_machine.user] = {
                'prob': int(users_probs[i] * 100),
                'time': sum([s['time'] for s in schedules[i]]),
                'list': [s for s in schedules[i]],
            }

        result['team_time'] = sum([result['schedule'][k]['time'] for k in result['schedule'].keys()])
        result['team_prob'] = sum([result['schedule'][k]['prob'] for k in result['schedule'].keys()])

        responses.append(result)

    return json.dumps(responses)


def longest_processing_time_first(m_machines=None, n_slots=None, save=False, id=None):

    m_machines_arrays = np.array([[{'time': 0}] * len(n_slots) for _ in range(len(m_machines))])
    n_slots_sorted = sorted(n_slots, key=lambda x: x.time, reverse=True)

    if save:
        # Remove all Slots relations
        [m.slots.clear() for m in m_machines]


    for i, n_slot in enumerate(n_slots_sorted):
        for j, m in enumerate(m_machines):
            if m.user == n_slot.user and n_slot.keep:
                m_machines_arrays[j][i] = {
                    'time': int(n_slot.time),
                    'index': n_slot.index,
                    'current': n_slot.index == id
                }

                if save:
                    m_machines[j].slots.add(n_slot)
                    n_slot.index = i
                    n_slot.user = n_slot.user
                    n_slot.save()


    for i, n_slot in enumerate(n_slots_sorted):
        if not n_slot.keep:
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