import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import datetime

from tqdm import tqdm

from Schedule_classes import TreeNode
from Schedule_analyse_dataset import default_dist

matplotlib.use('Agg')


def build():

    df = pd.read_csv('bunch_of_tasks_but_better.csv')
    df['reaction_timestamp'] = pd.to_datetime(df['reaction_timestamp'])

    tmp = df[df.reaction_timestamp > datetime.datetime.strptime('2016-06-01', '%Y-%m-%d')]
    tmp = tmp[-400000:]

    df = df[df.id.isin(tmp.id.values)]

    # Initialize the root node of the tree
    root = TreeNode('root')
    root.set_unique_placements(df['current_placement'].unique())

    discard = 0

    print("Build the tree")

    # Iterate over the records grouped by 'id'
    for _, group in tqdm(df.groupby('id')):

        prev_task = root

        if len(group) > 20:
            discard += 1

        for _, row in group[:21].iterrows():

            task_type = row['task_type']
            task_placement = row['current_placement']

            if task_type != 'created':
                task_type = f"{task_type}@{task_placement}"

            if task_type not in prev_task.children:
                prev_task.add_child(task_type)
            else:
                prev_task.count_child(task_type)

            prev_task.add_id(task_type, row.id)
            prev_task.add_reaction_time(task_type, row.reaction_time)
            prev_task.add_time_consumption(task_type, row.duration)

            prev_task = prev_task.get_child(task_type)

        prev_task.children['end'] = TreeNode('end')


    loc_time_consumption, scale_time_consumption, loc_reaction_time, scale_reaction_time = default_dist()

    for _, child_node in root.children.items():
        child_node.set_probabilities()
        child_node.set_distributions(
            time_consumption_default_loc=loc_time_consumption,
            time_consumption_default_scale=scale_time_consumption,
            reaction_time_default_loc=loc_reaction_time,
            reaction_time_default_scale=scale_reaction_time,
            time_consumption_default_dist_limit=100,
            reaction_time_default_dist_limit=100
        )

    return root


def visualize_tree(tree_root):
    def add_edges(tree_node, graph, level):
        for child_task_type, child_node in tree_node.children.items():
            edge_label = f"{child_node.count}"
            graph.add_edge(f'{({level - 1})} {tree_node.task_type}', f'{({level})} {child_task_type}', label=edge_label)
            add_edges(child_node, graph, level + 1)

    G = nx.DiGraph()
    add_edges(tree_root, G, 1)

    pos = nx.drawing.nx_pydot.pydot_layout(G, prog='dot')
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(100, 100))
    nx.draw(G, pos, node_color='lightblue', with_labels=True, arrows=True, font_size=12, node_size=4000, font_color='black', font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color='red', font_weight='bold')
    plt.savefig('tree_visualization.png')