
import numpy as np
import scipy.stats as stats
import sys

sys.setrecursionlimit(10000)


class TreeNode:

    def __init__(self, task_type):
        self.task_type = task_type
        self.probabilities = {}
        self.children = {}
        self.count = 1
        self.ids = []
        self.reaction_times = []
        self.time_consumptions = []
        self.time_consumption_loc = None
        self.time_consumption_scale = None
        self.reaction_time_loc = None
        self.reaction_time_scale = None
        self.dependent_job = None
        self.unique_placements = None

    def add_child(self, task_type):
        if task_type not in self.children:
            self.children[task_type] = TreeNode(task_type)

    def add_id(self, task_type, id):
        self.children.get(task_type).ids.append(id)

    def add_reaction_time(self, task_type, reaction_time):
        if reaction_time < 0:
            # print('Warning: reaction time less than 0')
            pass
        else:
            self.children.get(task_type).reaction_times.append(reaction_time)

    def add_time_consumption(self, task_type, time_consumption):
        if time_consumption < 0:
            # print('Warning: time consumption less than 0')
            pass
        else:
            self.children.get(task_type).time_consumptions.append(time_consumption)

    def get_child(self, task_type):
        return self.children.get(task_type)

    def count_child(self, task_type):
        self.children.get(task_type).count += 1

    def set_unique_placements(self, placements):
        self.unique_placements = placements

    def set_probabilities(self):
        count = 0
        for task_type, child_node in self.children.items():
            count += child_node.count
        for task_type, child_node in self.children.items():
            self.probabilities[task_type] = child_node.count / count
        for task_type, child_node in self.children.items():
            if task_type != 'end':
                child_node.set_probabilities()

    def set_distributions(
            self,
            time_consumption_default_loc=None,
            time_consumption_default_scale=None,
            time_consumption_default_dist_limit=100,
            reaction_time_default_loc=None,
            reaction_time_default_scale=None,
            reaction_time_default_dist_limit=100,
    ):
        if self.task_type != 'created':
            if any(x <= 0 for x in self.time_consumptions) or len(self.time_consumptions) < time_consumption_default_dist_limit:
                # print("Error for time_consumptions: using default distribution.")
                self.time_consumption_loc = time_consumption_default_loc
                self.time_consumption_scale = time_consumption_default_scale
            else:
                self.time_consumption_loc, self.time_consumption_scale = stats.expon.fit(self.time_consumptions)
            if any(x <= 0 for x in self.reaction_times) or len(self.reaction_times) < reaction_time_default_dist_limit:
                # print("Error for reaction_times: using default distribution.")
                self.reaction_time_loc = reaction_time_default_loc
                self.reaction_time_scale = reaction_time_default_scale
            else:
                self.reaction_time_loc, self.reaction_time_scale = stats.expon.fit(self.reaction_times)

        for task_type, child_node in self.children.items():
            if task_type != 'end':
                child_node.set_distributions(
                    time_consumption_default_loc,
                    time_consumption_default_scale,
                    time_consumption_default_dist_limit,
                    reaction_time_default_loc,
                    reaction_time_default_scale,
                    reaction_time_default_dist_limit
                )

    def sample(self, release_time):

        # Use time_consumption_left to sort by LONGEST / SHORTEST TIME LEFT
        # Use job_time_start to sort by FIFO

        end = False
        job = Job('created', 'unset', release_time=release_time, time_consumption=0, time_consumption_left=0)
        job_next = job
        task = self.get_child('created').get_next_event()
        while not end:
            if task.task_type == 'end':
                end = True
            else:
                release_time, time_consumption = task.get_dist()
                event, placement = task.task_type.split('@', 1)
                job_next.dependent_job = Job(event, placement, release_time=release_time, time_consumption=time_consumption, time_consumption_left=job_next.time_consumption_left + time_consumption)
                job_next = job_next.dependent_job
                task = task.get_next_event()
        return job



    def get_next_event(self):
        _, children = zip(*self.children.items())
        _, probabilities = zip(*self.probabilities.items())
        return np.random.choice(children, p=probabilities)

    def get_dist(self):

        if not self.time_consumption_scale:
            time_consumption_sample = [0.0]
        else:
            time_consumption_sample = stats.expon.rvs(loc=self.time_consumption_loc, scale=self.time_consumption_scale, size=1)

        if not self.reaction_time_scale:
            reaction_time_sample = [0.0]
        else:
            reaction_time_sample = stats.expon.rvs(loc=self.reaction_time_loc, scale=self.reaction_time_scale, size=1)

        return reaction_time_sample[0], time_consumption_sample[0]

    def __str__(self):
        return f"TreeNode({self.task_type})"

    def __repr__(self):
        return self.__str__()


class Job:

    def __init__(self, task, placement, release_time, time_consumption, time_consumption_left):
        self.task = task
        self.placement = placement
        self.release_time = release_time
        self.release_time_remaining = release_time
        self.release_time_actual = 0
        self.time_consumption = time_consumption
        self.time_consumption_left = time_consumption_left
        self.reaction_time = 0
        self.completion_time = 0
        self.dependent_job = None
        self.started = False

    def get_leaf(self):
        if self.dependent_job is None:
            return self
        else:
            return self.dependent_job.get_leaf()

    def __str__(self):
        return f"Job({self.task}@{self.placement})"

    def __repr__(self):
        return self.__str__()


class Machine:
    def __init__(self, placement, index):
        self.placement = placement
        self.index = index
        self.current_job = None
        self.time_remaining = 0