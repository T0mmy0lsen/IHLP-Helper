import pandas as pd
import datetime
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from scipy import stats
from tqdm import tqdm


def explore():

    df = pd.read_csv('bunch_of_tasks_but_better.csv', nrows=500000)
    df['reaction_timestamp'] = pd.to_datetime(df['reaction_timestamp'])

    tmp = df[df.reaction_timestamp > datetime.datetime.strptime('2016-06-01', '%Y-%m-%d')]

    df = df[~df.id.isin(tmp.id.values)]

    data = []

    # Iterate over the records grouped by 'id'
    for _, group in tqdm(df.groupby('id')):
        data.extend([e for e in group.duration.values if e >= 0])

    # Calculate basic statistics
    mean = np.mean(data)
    median = np.median(data)
    standard_deviation = np.std(data)
    variance = np.var(data)
    range_min = np.min(data)
    range_max = np.max(data)

    print("Mean:", mean)
    print("Median:", median)
    print("Standard Deviation:", standard_deviation)
    print("Variance:", variance)
    print("Range:", (range_min, range_max))

    # Plot histogram
    plt.hist(data, bins=50, alpha=0.75, density=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')
    plt.grid(True)
    plt.show()

    # Fit different distributions
    distributions = [stats.expon, stats.gamma, stats.norm, stats.lognorm, stats.weibull_min]
    fit_results = []

    for distribution in distributions:
        try:
            params = distribution.fit(data)
            kstest = stats.kstest(data, distribution.cdf, args=params)
            fit_results.append((distribution.name, params, kstest))
        except Exception as e:
            print(f"Error fitting {distribution.name}: {e}")

    # Sort the distributions by the goodness of fit (lower p-value indicates better fit)
    fit_results.sort(key=lambda x: x[2].pvalue, reverse=True)

    # Print the results
    for dist_name, params, kstest in fit_results:
        print(f"{dist_name}: params = {params}, ks_stat = {kstest.statistic}, pvalue = {kstest.pvalue}")

    # Plot the best fitting distribution
    best_distribution = fit_results[0]
    dist = getattr(stats, best_distribution[0])
    params = best_distribution[1]

    x = np.linspace(range_min, range_max, 100)
    pdf_fitted = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

    plt.plot(x, pdf_fitted, label=best_distribution[0])
    plt.hist(data, bins=50, alpha=0.75, density=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Best Fitting Distribution')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def default_dist():

    df = pd.read_csv('bunch_of_tasks_but_better.csv')
    df['reaction_timestamp'] = pd.to_datetime(df['reaction_timestamp'])

    tmp = df[df.reaction_timestamp > datetime.datetime.strptime('2016-06-01', '%Y-%m-%d')]
    tmp = tmp[-400000:]

    df = df[df.id.isin(tmp.id.values)]

    data_time_consumption = []
    data_reaction_time = []

    print("Build default distributions")

    # Iterate over the records grouped by 'id'
    for _, group in tqdm(df.groupby('id')):
        data_time_consumption.extend([e for e in group.duration.values if e >= 0])
        data_reaction_time.extend([e for e in group.reaction_time.values if e >= 0])

    loc_time_consumption, scale_time_consumption = stats.expon.fit(data_time_consumption)
    loc_reaction_time, scale_reaction_time = stats.expon.fit(data_reaction_time)

    return loc_time_consumption, scale_time_consumption, loc_reaction_time, scale_reaction_time


def test():
    loc_time_consumption, scale_time_consumption, loc_reaction_time, scale_reaction_time = default_dist()

    time_consumption_sample = stats.expon.rvs(loc=loc_time_consumption, scale=scale_time_consumption, size=10)
    reaction_time_sample = stats.expon.rvs(loc=loc_reaction_time, scale=scale_reaction_time, size=10)

    print(time_consumption_sample)
    print(reaction_time_sample)
