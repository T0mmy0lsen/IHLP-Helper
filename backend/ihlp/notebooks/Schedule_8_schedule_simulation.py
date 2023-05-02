
import random
from time import sleep

from tqdm import tqdm
from collections import deque

from ihlp.notebooks.Schedule_classes import Machine, Job
from ihlp.notebooks.Schedule_7_3_distributions_schedule_preprocess import build

import datetime
import pandas as pd
import numpy as np
import copy
import threading
import sys
import time

sys.setrecursionlimit(100000)

# A task is created every X min.
job_creation_interval = 1

np.random.seed(42)

start_time = time.time()


class Stats:

    job_popped = 0
    job_assigned = 0
    job_spawned = 0
    job_instant = 0
    job_released = 0
    job_completed = 0
    job_consumption = 0
    job_consumption_expected = 0
    job_reaction_time_reduction = 0

    job_completed_queue = {}
    job_released_queue = {}
    job_idle_count = {}

    schedule_makespan = 0

    def __str__(self):
        completed_queue = ', '.join(f"{k}: {v}" for k, v in self.job_completed_queue.items())

        return (
            f"Stats:\n"
            f"  Released {self.job_released} jobs and spawned {self.job_spawned} jobs.\n"    
            f"  All released jobs are instantly freed, since they are just dummy jobs to spawn jobs using the TreeNode.\n"    
            f"  Popped {self.job_popped - self.job_released} jobs, with {self.job_assigned} being assigned to machines, the last {self.job_instant - self.job_released} jobs was instantly freed.\n"
            f"  The difference in the spawned jobs and the popped jobs are the jobs in the queue and the jobs awaiting to be released to the queue.\n"
            f"  Completed {self.job_completed} jobs, which means {self.job_popped - self.job_released - (self.job_instant - self.job_released) - self.job_completed} is expected to be waiting in the machines.\n"
            f"  Consumption {self.job_consumption} of processing time.\n"
            f"  Consumption if all machines completed their current jobs {self.job_consumption_expected}\n"
            f"  Number of minutes machines where idle: {self.job_idle_count}\n"
            f"  Number of minutes Reaction Time was reduced: {self.job_reaction_time_reduction}\n"
            f"  Completed Queue: {{{completed_queue}}}\n"
            f"  Schedule Makespan: {self.schedule_makespan}\n"
        )


def unfold(job):
    arr = []
    while job.dependent_job:
        arr.append(job)
        job = job.dependent_job
    return arr


def sort_by(queue, method):
    if method == 'FIFO':
        return queue
    if method == 'SHORTEST-TIME-LEFT':
        return deque(sorted(queue, key=lambda j: j.time_consumption_left))
    if method == 'EARLIEST-DEADLINE':
        return deque(sorted(queue, key=lambda j: j.release_time))
    if method == 'DYNAMIC_001':
        return deque(sorted(queue, key=lambda j: ((60 * 24 * 3) - j.release_time ) * (1/(60 * 24 * 1)) + j.time_consumption_left))
    if method == 'DYNAMIC_025':
        return deque(sorted(queue, key=lambda j: ((60 * 24 * 3) - j.release_time ) * (1/(60 * 24 * 25)) + j.time_consumption_left))
    if method == 'DYNAMIC_075':
        return deque(sorted(queue, key=lambda j: ((60 * 24 * 3) - j.release_time ) * (1/(60 * 24 * 75)) + j.time_consumption_left))
    if method == 'DYNAMIC_150':
        return deque(sorted(queue, key=lambda j: ((60 * 24 * 3) - j.release_time ) * (1/(60 * 24 * 150)) + j.time_consumption_left))
    if method == 'DYNAMIC_300':
        return deque(sorted(queue, key=lambda j: ((60 * 24 * 3) - j.release_time ) * (1/(60 * 24 * 300)) + j.time_consumption_left))


def random_inactive_period():
    # Choose a time unit (minutes, hours, days, or weeks)
    time_unit = np.random.choice(['minutes', 'hours', 'days', 'weeks'], p=[0.8, 0.17, 0.02, 0.01])

    if time_unit == 'minutes':
        return np.random.randint(1, 60)
    elif time_unit == 'hours':
        return np.random.randint(1, 24) * 60
    elif time_unit == 'days':
        return np.random.randint(1, 7) * 24 * 60
    elif time_unit == 'weeks':
        return np.random.randint(1, 2) * 7 * 24 * 60


def generate_active_periods(num_machines, total_simulation_time):
    active_periods = []

    for _ in range(num_machines):

        machine_schedule = np.zeros(total_simulation_time, dtype=int)
        current_time = 0

        while current_time < total_simulation_time:

            if np.random.randint(1, 100) <= 1:
                # Generate random inactive period
                inactive_period = random_inactive_period()
                current_time += inactive_period
            else:
                machine_schedule[current_time] = 1

            if current_time >= total_simulation_time:
                break

            current_time += 1

        active_periods.append(machine_schedule.tolist())

    return active_periods


def simulate_online_scheduling(
        jobs=None,
        unique_placements=None,
        total_simulation_time=0,
        complete=True,
        scale=1,
        method='FIFO',
        machines_hours=None,
        machines_hours_active=False,
):

    machines = []
    for i in range(scale):
        machines += [Machine(placement, i) for placement in unique_placements]

    machines_extra = {
        'created': 4,
        'unset': 4,
    }

    for k, v in machines_extra.items():
        for i in range(v):
            machines.append(Machine(k, i + scale))

    stats = Stats()
    queue = {}
    queue_for_awaiting_release = jobs
    total_simulation_time = int(total_simulation_time)

    for machine in machines:
        queue[machine.placement] = deque()
        stats.job_completed_queue[machine.placement] = 0
        stats.job_idle_count[machine.placement] = 0

    def run(s, current_time):
        for machine in machines:
            # If the machine is done processing, we find a new job for it.
            if machine.time_remaining <= 0:
                if len(queue[machine.placement]) > 0:
                    # Sort the queue according to the scheduling algorithm.
                    queue[machine.placement] = sort_by(queue[machine.placement], method)

                    next_job = queue[machine.placement].popleft()
                    if next_job.task == 'created':
                        next_job.reaction_time = current_time - next_job.release_time_actual

                    machine.current_job = next_job
                    machine.current_job.started = True
                    machine.time_remaining = int(next_job.time_consumption)
                else:
                    # There are no jobs in the queue. Leave the machine idle.
                    s.job_idle_count[machine.placement] += 1

            if machine.time_remaining > 0:
                decrement_time = machine.placement in ['unset', 'created'] or not machines_hours_active or (machines_hours_active and machines_hours[machine.placement][machine.index][current_time] == 1)
                if decrement_time:
                    # Decrement time remaining on the current job
                    machine.time_remaining -= 1

            if machine.time_remaining <= 0 and machine.current_job is not None:
                machine.current_job.completion_time = current_time
                s.job_completed_queue[machine.placement] += 1
                if machine.current_job.dependent_job:
                    queue_for_awaiting_release.append(machine.current_job.dependent_job)
                machine.current_job = None


    def release(current_time):
        subtract = 960 if current_time % 960 == 959 else 1
        released_jobs = []

        for job in queue_for_awaiting_release:
            job.release_time_remaining -= subtract
            if job.release_time_remaining <= 0:
                job.release_time_actual = current_time
                if job.placement in queue:
                    queue[job.placement].append(job)
                released_jobs.append(job)

        queue_for_awaiting_release[:] = [job for job in queue_for_awaiting_release if job not in released_jobs]


    current_time = 0
    print(f"Start iteration: {time.time() - start_time}")

    for _ in tqdm(range(total_simulation_time)):
        release(current_time)
        queue_for_awaiting_release = [e for e in queue_for_awaiting_release if e.release_time_remaining > 0]
        run(stats, current_time)
        current_time += 1

    if complete:
        while sum([len(v) for k, v in queue.items()]) > 0 or len(queue_for_awaiting_release) > 0:
            release(current_time)
            queue_for_awaiting_release = [e for e in queue_for_awaiting_release if e.release_time_remaining > 0]
            run(stats, current_time)
            current_time += 1

    return stats, queue, queue_for_awaiting_release, jobs


def create_job_queue(cross_validate=0, days=30):

    df = pd.read_csv('bunch_of_tasks_but_better.csv')
    df = df.fillna('')
    df = df[df.current_placement != '']
    df['reaction_timestamp'] = pd.to_datetime(df['reaction_timestamp'])

    date_y = pd.to_datetime('2022-12-31') - pd.DateOffset(days=(cross_validate * days))
    date_x = pd.to_datetime('2022-12-31') - pd.DateOffset(days=(cross_validate * days) + days)

    tmp = df[(df['event'] == 'created') & (df['reaction_timestamp'] >= date_x) & (df['reaction_timestamp'] <= date_y)]

    df = df[df.id.isin(tmp.id.values) & (df.reaction_time < 129600)]
    df = df.sort_values(by='reaction_timestamp')

    print(len(df))

    df['reaction_timestamp'] = (df['reaction_timestamp'] - date_x).dt.total_seconds() / 60
    df['duration'] = df['duration'] + 1

    jobs = []

    for _, group in df.groupby('id'):
        last_job = None
        time_consumption_left = sum(group.duration.values)
        for _, row in group.iterrows():

            job = Job(
                task=row.event,
                placement=row.current_placement,
                release_time=row.reaction_time,
                time_consumption=row.duration,
                time_consumption_left=time_consumption_left
            )

            if not last_job:
                job.release_time = row.reaction_timestamp
                job.release_time_remaining = row.reaction_timestamp
                job.placement = 'unset'
                jobs.append(job)
            else:
                last_job.dependent_job = job
            last_job = job
            time_consumption_left -= row.duration

    unique_placements = np.unique(df.current_placement.values)

    return jobs, unique_placements


root = build()
print("Starting simulation")
sleep(1)


def copy_shift(e, c):
    tmp = copy.deepcopy(e)
    tmp.release_time_remaining -= c
    return tmp


def create_job_extra(_root, _max_time_step):

    jobs = []
    for i in range(30):
        rand = random.randint(1, _max_time_step)
        for _ in range(random.randint(1, 25)):
            sample = root.sample(rand)
            jobs.append(sample)

    return jobs


for cross_validate in [0, 1, 2, 3, 4]:
    jobs, unique_placements = create_job_queue(cross_validate, days=30)
    max_time_step = int(jobs[-1].release_time_remaining)
    jobs_extra = create_job_extra(_root=root, _max_time_step=max_time_step)
    machines_hours = {placement: generate_active_periods(3, max_time_step) for placement in unique_placements}
    for e in ['FIFO', 'SHORTEST-TIME-LEFT', 'EARLIEST-DEADLINE', 'DYNAMIC_001', 'DYNAMIC_025', 'DYNAMIC_075', 'DYNAMIC_150', 'DYNAMIC_300']:
        for horizontal_scaling in [1, 2, 3]:
            for vertical_scaling in [1, 2, 3]:
                for jobs_extra_active in [True, False]:
                    for machines_hours_active in [True, False]:

                        print(f"Start simulation: {time.time() - start_time}")

                        jobs_copy = []

                        if jobs_extra_active:
                            jobs_copy += [copy_shift(e, 0) for e in jobs_extra]
                        for c in range(vertical_scaling):
                            jobs_copy += [copy_shift(e, c - 1) for e in jobs]

                        stats, queue, queue_for_awaiting_release, all_jobs = simulate_online_scheduling(
                            jobs=jobs_copy,
                            unique_placements=unique_placements,
                            total_simulation_time=max_time_step,
                            complete=False,
                            scale=horizontal_scaling,
                            method=e,
                            machines_hours=machines_hours,
                            machines_hours_active=machines_hours_active,
                        )

                        print(f"Start completed: {time.time() - start_time}")

                        reaction_time = 0
                        completion_time = 0
                        deadline_time = 0

                        all_jobs = [e for e in all_jobs if e.started and e.get_leaf().completion_time > 0]

                        for job in all_jobs:
                            reaction_time = job.reaction_time
                            release_time = job.release_time_actual
                            completion_time += job.get_leaf().completion_time - release_time - reaction_time
                            reaction_time += reaction_time
                            deadline = (60 * 24 * 3) - (job.get_leaf().completion_time - release_time)
                            deadline_time += deadline if deadline >= 0 else 0

                        queue_size = sum([len(v) for k, v in queue.items()])
                        idle_time = sum([v for k, v in stats.job_idle_count.items()])

                        print(f"Write to: results_{cross_validate}.csv")
                        with open(f"results_{cross_validate}.csv", 'a') as file:
                            file.write(
                                f"{cross_validate},"
                                f"{e},"
                                f"{horizontal_scaling},"
                                f"{vertical_scaling},"
                                f"{len(queue_for_awaiting_release)},"
                                f"{queue_size},"
                                f"{idle_time},"
                                f"{(completion_time / len(all_jobs))},"
                                f"{(reaction_time / len(all_jobs))},"
                                f"{(deadline_time / len(all_jobs))},"
                                f"{machines_hours_active},"
                                f"{jobs_extra_active}"
                                f"\n"
                            )