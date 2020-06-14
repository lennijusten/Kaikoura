# PickVisualization
# Author: Lennart Justen
# Last revision: 6/13/20

# Description:

import obspy
import numpy as np
import os
import pandas as pd
import shlex
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools
import statistics
from scipy import stats

PN_pick_method = ['earliest']
outlier_method = ['over', 2]

dataset_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset'
output_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/PhaseNet/output'
arrival_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/Arrival.pickle'

headers = ['network', 'event_id', 'station', 'channel', 'samples', 'delta', 'start', 'end', 'P_residual', 'P_time',
           'P_phasenet', 'tp_prob', 'itp', 'S_residual', 'S_time', 'S_phasenet', 'ts_prob', 'its', 'fname']

# METHODS:
# earliest -- Uses PhaseNet's earliest pick as the arrival pick
# max_prob -- Uses the PhaseNet pick with the highest probability as the arrival pick

methods = {
    "earliest": 'Uses PhaseNets earliest pick as the arrival pick.',
    "max_prob": 'Uses the PhaseNet pick with the highest probability as the arrival pick.',
    "IQR": 'IQR outliers excluded',
    "over": 'outliers over a limit excluded'
}


def csvSync(dataset, output, arrival, sorted_headers, method):
    log = pd.read_csv(os.path.join(dataset, 'data_log.csv'))
    picks = pd.read_csv(os.path.join(output, 'picks.csv'))
    arrivals = pd.read_pickle(arrival)
    arrivals = arrivals.drop(columns=['error_x', 'method_x', 'error_y', 'method_y'], axis=1)

    print("\nMerging picks.csv with data_log.csv...")
    df = pd.merge(log, picks, how='left', on=['fname'])

    print("Reformatting data from csv files...")
    for col in ['start', 'end']:
        df[col] = [obspy.UTCDateTime(x) for x in df[col]]

    for col2 in ['itp', 'its']:
        a = []
        for x in range(len(df[col2])):
            try:
                a.append(list(map(int, shlex.split(df[col2][x].strip('[]')))))
            except AttributeError:
                a.append([])
                # print("Pick sample data is already in the correct format. Passing")
                pass
        df[col2] = a

    for col3 in ['tp_prob', 'ts_prob']:
        b = []
        for x in range(len(df[col3])):
            try:
                b.append(list(map(float, shlex.split(df[col3][x].strip('[]')))))
            except AttributeError:
                b.append([])
                # print("Pick probability data is already in the correct format. Passing")
                pass
        df[col3] = b

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

    p_diffs = []
    p_diff = []
    s_diffs = []
    s_diff = []
    p_prob_list = []
    s_prob_list = []
    p_empty_count = 0
    s_empty_count = 0
    for row2 in range(len(df['p_time'])):
        ptimes = df['p_time'][row2]
        stimes = df['s_time'][row2]
        pdiff_lst, sdiff_lst = [], []
        for pt in ptimes:
            try:
                pdiff_lst.append(df['P_time'][row2] - pt)
            except TypeError:
                pass
        p_diffs.append(pdiff_lst)
        for st in stimes:
            try:
                sdiff_lst.append(df['S_time'][row2] - st)
            except TypeError:
                pass
        s_diffs.append(sdiff_lst)
        if method == 'earliest':
            try:
                if df['itp'][row2][0] != 1:
                    p_diff.append(pdiff_lst[0])
                    p_prob_list.append(df['tp_prob'][row2][0])
                else:
                    p_diff.append(pdiff_lst[1])
                    p_prob_list.append(df['tp_prob'][row2][1])
            except IndexError:
                p_empty_count += 1
                p_diff.append(np.nan)
                p_prob_list.append(np.nan)
            try:
                if df['its'][row2][0] != 1:
                    s_diff.append(sdiff_lst[0])
                    s_prob_list.append(df['ts_prob'][row2][0])
                else:
                    s_diff.append(sdiff_lst[1])
                    s_prob_list.append(df['ts_prob'][row2][1])
            except IndexError:
                s_empty_count += 1
                s_diff.append(np.nan)
                s_prob_list.append(np.nan)
        elif method == 'max_prob':
            try:
                p_diff.append(pdiff_lst[df['tp_prob'][row2].index(max(df['tp_prob'][row2]))])
                p_prob_list.append(max(df['tp_prob'][row2]))
            except ValueError:
                p_empty_count += 1
                p_diff.append(np.nan)
                p_prob_list.append(np.nan)
            except IndexError:
                s_empty_count += 1
                s_diff.append(np.nan)
                s_prob_list.append(np.nan)
            try:
                s_diff.append(sdiff_lst[df['ts_prob'][row2].index(max(df['ts_prob'][row2]))])
                s_prob_list.append(max(df['ts_prob'][row2]))
            except ValueError:
                s_empty_count += 1
                s_diff.append(np.nan)
                s_prob_list.append(np.nan)
            except IndexError:
                s_empty_count += 1
                s_diff.append(np.nan)
                s_prob_list.append(np.nan)
        else:
            print("Invalid method: method = (['earliest'], ['max_prob')]")

    df['P_residual'] = p_diffs
    df['S_residual'] = s_diffs

    df = df.rename(columns={"p_time": "P_phasenet", "s_time": "S_phasenet"})
    df = df[sorted_headers].sort_values(['event_id', 'station'])
    df.to_pickle(os.path.join(dataset, "data_log_merged.pickle"))
    df.to_csv(os.path.join(dataset, "data_log_merged.csv"), index=False)
    print("Merge successful. Copying files to ", dataset)
    return df, p_diff, s_diff, p_prob_list, s_prob_list, p_empty_count, s_empty_count


df2, p_res, s_res, p_prob, s_prob, p_empty, s_empty = csvSync(dataset_path, output_path, arrival_path, headers,
                                                              PN_pick_method[0])

pssr = np.nansum([i ** 2 for i in p_res])
sssr = np.nansum([i ** 2 for i in s_res])
print("P-SSR = ", pssr)
print("S-SSR - ", sssr)


def outliers(p_residuals, s_residuals, p_probability, s_probability, method):
    p_data = np.array(p_residuals)[~np.isnan(p_residuals)]
    s_data = np.array(s_residuals)[~np.isnan(s_residuals)]
    p_prob_data = np.array(p_probability)[~np.isnan(p_probability)]
    s_prob_data = np.array(s_probability)[~np.isnan(s_probability)]

    if method[0] == 'IQR:':
        pq1, pq3 = np.percentile(p_data, [25, 75])
        sq1, sq3 = np.percentile(s_data, [25, 75])

        piqr = pq3 - pq1
        siqr = sq3 - sq1

        p_lower_bound = pq1 - (1.5 * piqr)
        p_upper_bound = pq3 + (1.5 * piqr)
        s_lower_bound = sq1 - (1.5 * siqr)
        s_upper_bound = sq3 + (1.5 * siqr)
    elif method[0] == 'over':
        p_lower_bound = -method[1]
        p_upper_bound = method[1]
        s_lower_bound = -method[1]
        s_upper_bound = method[1]
    else:
        print("Invalid outlier method: method = (['IQR'], ['over', limit])")

    print("S Upper bound ({} method): {}".format(method[0], s_upper_bound))
    print("P Lower bound ({} method): {}".format(method[0], p_lower_bound))
    print("P Upper bound ({} method): {}".format(method[0], p_upper_bound))
    print("S Lower bound ({} method): {}".format(method[0], s_lower_bound))

    p_inrange_idx = list(itertools.chain.from_iterable(np.array(
        np.where(np.logical_and(p_data >= p_lower_bound, p_data <= p_upper_bound))).tolist()))
    p_inrange = p_data[p_inrange_idx]
    p_inrange_probs = p_prob_data[p_inrange_idx]
    p_outliers_idx = list(itertools.chain.from_iterable(np.array(
        np.where(np.logical_or(p_data < p_lower_bound, p_data > p_upper_bound))).tolist()))
    p_outliers = p_data[p_outliers_idx]
    p_outlier_probs = p_prob_data[p_outliers_idx]
    print("{} P outliers found ({} in range)".format(len(p_outliers), len(p_inrange)))

    s_inrange_idx = list(itertools.chain.from_iterable(np.array(
        np.where(np.logical_and(s_data >= s_lower_bound, s_data <= s_upper_bound))).tolist()))
    s_inrange = s_data[s_inrange_idx]
    s_inrange_probs = s_prob_data[s_inrange_idx]
    s_outliers_idx = list(itertools.chain.from_iterable(np.array(
        np.where(np.logical_or(s_data < s_lower_bound, s_data > s_upper_bound))).tolist()))
    s_outliers = s_data[s_outliers_idx]
    s_outlier_probs = s_prob_data[s_outliers_idx]
    print("{} S outliers found ({} in range)".format(len(s_outliers), len(s_inrange)))

    return p_inrange, s_inrange, p_inrange_probs, s_inrange_probs, len(p_outliers), len(s_outliers)


P, S, Pp, Sp, P_out, S_out = outliers(p_res, s_res, p_prob, s_prob, outlier_method)


def Histogram(res, outliers, phase, Methods, method):
    fig, ax = plt.subplots()
    if phase == 'P':
        c = '#0504aa'
    elif phase == 'S':
        c = '#ff7f0e'
    else:
        print("Invalid phase entry: phase= ('P', 'S')")

    plt.xlim(-method[1], method[1])

    n, bins, patches = plt.hist(x=res, bins='auto', color=c,
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Residual time (s)')
    plt.ylabel('Frequency*')
    plt.title('Histogram of {}-pick Residuals (expected-observed)'.format(phase))
    textstr = '\n'.join((
        r'$n=%.0f$' % (len(res),),
        r'$\mu=%.4f$' % (statistics.mean(res),),
        r'$\mathrm{median}=%.4f$' % (round(statistics.median(res), 4),),
        r'$\sigma=%.4f$' % (statistics.pstdev(res),)))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='square,pad=.6', facecolor='lightgrey', edgecolor='black', alpha=1))
    plt.annotate('*method={}: {} '.format(method, Methods[method[0]]), (0, 0), (0, -40),
                 xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
    plt.annotate('{} outliers removed'.format(outliers), (0, 0), (0, -50),
                 xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


Histogram(P, P_out, 'P', methods, outlier_method)
Histogram(S, S_out, 'S', methods, outlier_method)


def scatterPlot(P, S, P_prob, S_prob, P_out, S_out, Methods, method):
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.concatenate((abs(P), abs(S)), axis=None),
                                                                   np.concatenate((P_prob, S_prob), axis=None))
    line, = plt.plot(np.concatenate((abs(P), abs(S)), axis=None),
                     intercept + slope * np.concatenate((abs(P), abs(S)), axis=None), "r--")
    pscat = plt.scatter(abs(P), P_prob)
    pscat.set_label('P')
    sscat = plt.scatter(abs(S), S_prob)
    sscat.set_label('S')
    plt.grid(axis='both', alpha=0.4)
    plt.xlim(0, method[1])
    plt.xlabel('abs(Residual time) (s)')
    plt.ylabel('PhaseNet probability*')
    plt.title('Scatterplot of P & S Residuals by PhaseNet Probability, $r^2={}$'.format(round(r_value ** 2, 3)))

    fc = colors.to_rgba('lightgrey')
    ec = colors.to_rgba('black')
    fc = fc[:-1] + (0.7,)
    plt.annotate("y=%.3fx+%.3f" % (slope, intercept), xy=(.715, .99), xytext=(12, -12), va='top',
                 xycoords='axes fraction', textcoords='offset points',
                 bbox=dict(facecolor=fc, edgecolor=ec, boxstyle='square,pad=.6'))
    plt.annotate('*method={}: {} '.format(method, Methods[method[0]]), (0, 0), (0, -40),
                 xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
    plt.annotate('{} outliers removed'.format(P_out + S_out), (0, 0), (0, -50),
                 xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
    plt.legend(loc=(.875, .70))
    print("y=%.6fx+(%.6f)" % (slope, intercept))
    plt.show()


scatterPlot(P, S, Pp, Sp, P_out, S_out, methods, outlier_method)
