import numpy as np

_data = [
    {'time': 1},
    {'time': 2},
    {'time': 4},
    {'time': 10},
    {'time': 12},
    {'time': 32},
    {'time': 55},
    {'time': 92},
]

def longest_processing_time_first(m_machines, n_jobs):
    m_machines_arrays = np.array([[0] * len(n_jobs) for i in range(m_machines)])
    n_jobs_sorted = sorted(n_jobs, key=lambda x: x['time'], reverse=True)
    for i, n_job in enumerate(n_jobs_sorted):
        index_with_lowest_sum = np.argmin(np.array(m_machines_arrays).sum(axis=1))
        m_machines_arrays[index_with_lowest_sum][i] = n_job['time']
    return m_machines_arrays

result = longest_processing_time_first(4, _data)
print(result)