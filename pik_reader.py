import obspy
import numpy as np
import os
import csv
import pandas as pd
import shlex
import matplotlib.pyplot as plt
import itertools
import statistics
from scipy import stats

method = 'earliest'

dataset_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset'
output_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/PhaseNet/output'
arrival_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/Arrival.pickle'

headers = ['network', 'event_id', 'station', 'channel', 'samples', 'delta', 'start', 'end', 'residual', 'phase', 'time',
           'p_time', 'tp_prob', 's_time', 'ts_prob', 'itp', 'its', 'fname']

# METHODS:
# earliest -- Uses PhaseNet's earliest pick as the arrival pick
# max_prob -- Uses the PhaseNet pick with the highest probability as the arrival pick

methods = {
    "earliest": 'Uses PhaseNets earliest pick as the arrival pick.',
    "max_prob": 'Uses the PhaseNet pick with the highest probability as the arrival pick.'
}


def csvSync(dataset, output, arrival, sorted_headers, method):
    diff_method = method
    empty_count = 0

    log = pd.read_csv(os.path.join(dataset, 'data_log.csv'))
    picks = pd.read_csv(os.path.join(output, 'picks.csv'))
    arrivals = pd.read_pickle(arrival)
    arrivals = arrivals.drop(columns=['error', 'method'], axis=1)

    arrivals['channel'] = [x.replace(x[len(x) - 1], '?') for x in arrivals['channel']]

    print("\nMerging picks.csv with data_log.csv...")
    df = pd.merge(log, picks, how='left', on=['fname'])

    print("Reformatting data from csv files...")
    for col in ['start', 'end']:
        df[col] = [obspy.UTCDateTime(x) for x in df[col]]

    for col2 in ['itp', 'its']:
        try:
            df[col2] = [list(map(int, shlex.split(x.strip('[]')))) for x in df[col2]]
        except AttributeError:
            print("Pick sample data is already in the correct format. Passing")
            pass

    for col3 in ['tp_prob', 'ts_prob']:
        try:
            df[col3] = [list(map(float, shlex.split(x.strip('[]')))) for x in df[col3]]
        except AttributeError:
            print("Pick probability data is already in the correct format. Passing")
            pass

    utc_p_picks = []
    utc_s_picks = []
    for row in range(len(df['itp'])):
        p_lst = df['itp'][row]
        s_lst = df['its'][row]
        p_lst2, s_lst2 = [], []
        for p_element in p_lst:
            p_lst2.append(df['start'][row] + float(p_element) * df['delta'][row])
        for s_element in s_lst:
            s_lst2.append(df['start'][row] + float(s_element) * df['delta'][row])
        utc_p_picks.append(p_lst2)
        utc_s_picks.append(s_lst2)

    df['p_time'] = utc_p_picks
    df['s_time'] = utc_s_picks

    print("Merging with arrival.pickle...")
    df = pd.merge(df, arrivals, how='left', on=['event_id', 'station', 'network', 'channel'])

    diff_list = []
    diff = []
    prob_list = []
    for row2 in range(len(df['p_time'])):
        ptimes = df['p_time'][row2]
        stimes = df['s_time'][row2]
        pdiff_lst, sdiff_lst = [], []
        for pt in ptimes:
            pdiff_lst.append(df['time'][row2] - pt)
        diff_list.append(pdiff_lst)
        if method == 'earliest':
            try:
                diff.append(pdiff_lst[0])
                prob_list.append(df['tp_prob'][row2][0])
            except IndexError:
                empty_count += 1
                diff.append(np.nan)
                prob_list.append(np.nan)
        elif method == 'max_prob':
            try:
                diff.append(pdiff_lst[df['tp_prob'][row2].index(max(df['tp_prob'][row2]))])
                prob_list.append(max(df['tp_prob'][row2]))
            except ValueError:
                empty_count += 1
                diff.append(np.nan)
        else:
            print("Invalid method: method = ('earliest', 'max_prob')")

    df['residual'] = diff_list

    df = df[sorted_headers].sort_values(['event_id', 'station'])
    df.to_pickle(os.path.join(dataset, "data_log_merged.pickle"))
    df.to_csv(os.path.join(dataset, "data_log_merged.csv"), index=False)
    print("Merge successful. Copying files to ", dataset)
    return df, diff, prob_list, empty_count


df2, residual, prob, empty = csvSync(dataset_path, output_path, arrival_path, headers, method)
# clean_residual = np.array(residual)[~np.isnan(residual)]

res = np.nansum([i ** 2 for i in residual])
print("SSR = ", res)


def outliers(residuals, probs):
    data = np.array(residuals)[~np.isnan(residuals)]
    prob_data = np.array(probs)[~np.isnan(probs)]
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    print("Lower bound (IQR method): ", lower_bound)
    upper_bound = q3 + (1.5 * iqr)
    print("Upper bound (IQR method): ", upper_bound)

    inrange_idx = list(itertools.chain.from_iterable(np.array(
        np.where(np.logical_and(data >= lower_bound, data <= upper_bound))).tolist()))
    inrange = data[inrange_idx]
    inrange_probs = prob_data[inrange_idx]
    outliers_idx = list(itertools.chain.from_iterable(np.array(
        np.where(np.logical_or(data < lower_bound, data > upper_bound))).tolist()))
    outliers = data[outliers_idx]
    outlier_probs = prob_data[outliers_idx]
    print("{} outliers found ({} in range)".format(len(outliers), len(inrange)))
    return inrange, inrange_probs, len(outliers)


data, probs, n_outliers = outliers(residual, prob)


fig, ax = plt.subplots()
n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Residual time (s)')
plt.ylabel('Frequency*')
plt.title('Histogram of Residuals (expected-observed)')
textstr = '\n'.join((
    r'$n=%.0f$' % (len(data), ),
    r'$\mu=%.4f$' % (statistics.mean(data), ),
    r'$\mathrm{median}=%.4f$' % (round(statistics.median(data), 4), ),
    r'$\sigma=%.4f$' % (statistics.pstdev(data), )))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='square,pad=.6',facecolor='lightgrey', edgecolor='black', alpha=1))
plt.annotate('*method={}: {} '.format(method, methods[method]), (0,0), (0, -40),
             xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
plt.annotate('{} outliers excluded'.format(n_outliers), (0,0), (0, -50),
             xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

plt.scatter(abs(data), probs)
plt.grid(axis='both', alpha=0.4)
plt.xlabel('abs(Residual time) (s)')
plt.ylabel('PhaseNet probability*')
slope, intercept, r_value, p_value, std_err = stats.linregress(abs(data), probs)
plt.plot(abs(data), intercept + slope * abs(data), "r--")
plt.title('Scatterplot of Residuals by Probability, $r^2={}$'.format(round(r_value ** 2, 3)))
plt.annotate("y=%.3fx+%.3f" % (slope, intercept), xy=(.715, .99), xytext=(12, -12), va='top',
             xycoords='axes fraction', textcoords='offset points',
             bbox=dict(facecolor='lightgrey',alpha=1, edgecolor='black', boxstyle='square,pad=.6'))
plt.annotate('*method={}: {} '.format(method, methods[method]), (0,0), (0, -40),
             xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
plt.annotate('{} outliers excluded'.format(n_outliers), (0,0), (0, -50),
             xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
print("y=%.6fx+(%.6f)" % (slope, intercept))
plt.show()
