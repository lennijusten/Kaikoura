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
from matplotlib import gridspec
import itertools
import statistics
from scipy import stats

PN_pick_method = ['max_prob']
outlier_method = ['over', 2]

dataset_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset'
outlier_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset/Outliers'
output_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/PhaseNet/output'
arrival_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/Arrival.pickle'
plot_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset/Plots'

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


def initFrames(dataset_path, output_path, arrival_path):
    print("Initializing dataframes...")
    log = pd.read_csv(os.path.join(dataset_path, 'data_log.csv'))
    picks = pd.read_csv(os.path.join(output_path, 'picks.csv'))
    arrivals = pd.read_pickle(arrival_path)
    return log, picks, arrivals


log, picks, arrivals = initFrames(dataset_path, output_path, arrival_path)

df = pd.merge(log, picks, how='left', on=['fname'])
df = pd.merge(df, arrivals[["event_id", "station", "channel", "network", "P_time", "S_time"]],
              how='left', on=['event_id', 'station', 'network', 'channel'])


def typeConverter(df):
    print("Cleaning PhaseNet pick data...")
    for col in ['start', 'end']:
        df[col] = [obspy.UTCDateTime(x) for x in df[col]]

    for col2 in ['itp', 'its']:
        a = []
        for x in range(len(df)):
            try:
                a.append(list(map(int, shlex.split(df[col2][x].strip('[]')))))
            except AttributeError:
                a.append([])
                pass
        df[col2] = a

    for col3 in ['tp_prob', 'ts_prob']:
        b = []
        for x in range(len(df)):
            try:
                b.append(list(map(float, shlex.split(df[col3][x].strip('[]')))))
            except AttributeError:
                b.append([])
                pass
        df[col3] = b
    return df


df = typeConverter(df)


def pick2time(df):
    print("Converting PhaseNet picks into UTC times...")
    p_utc_picks = []
    s_utc_picks = []
    for row in range(len(df)):
        p_lst, s_lst = [], []
        for p_element in df['itp'][row]:
            p_lst.append(obspy.UTCDateTime(df['start'][row] + float(p_element) * df['delta'][row]))
        for s_element in df['its'][row]:
            s_lst.append(obspy.UTCDateTime(df['start'][row] + float(s_element) * df['delta'][row]))
        p_utc_picks.append(p_lst)
        s_utc_picks.append(s_lst)

    df['P_phasenet'] = p_utc_picks
    df['S_phasenet'] = s_utc_picks
    return df


df = pick2time(df)


def resCalculator(df):
    print("Calculating residuals from Geonet arrival times...")
    p_res_lst = []
    s_res_lst = []
    for row in range(len(df)):
        pdiffs, sdiffs = [], []
        for pt in df['P_phasenet'][row]:
            try:
                pdiffs.append(df['P_time'][row] - pt)
            except TypeError:
                pass
        p_res_lst.append(pdiffs)
        for st in df['S_phasenet'][row]:
            try:
                sdiffs.append(df['S_time'][row] - st)
            except TypeError:
                pass
        s_res_lst.append(sdiffs)

    df['P_residual'] = p_res_lst
    df['S_residual'] = s_res_lst
    return df


df = resCalculator(df)
df = df[headers].sort_values(['event_id', 'station'])
df.to_pickle(os.path.join(dataset_path, "data_log_merged.pickle"))
df.to_csv(os.path.join(dataset_path, "data_log_merged.csv"), index=False)
print("Success! Merged dataframe being saved to ", dataset_path)

print("=================================================================================")


def picker(df, method):
    print("Initializing pick algorithm... (method = {})".format(method))
    fname = []

    p_pick = []
    p_res = []
    p_prob = []
    p_empty_count = 0

    s_pick = []
    s_res = []
    s_prob = []
    s_empty_count = 0

    if method == 'earliest':
        for row in range(len(df)):
            fname.append(df['fname'][row])
            if not df['itp'][row]:
                p_pick.append(np.nan)
                p_res.append(np.nan)
                p_prob.append(np.nan)
                p_empty_count += 1
            elif df['itp'][row][0] != 1:
                p_pick.append(df['P_phasenet'][row][0])
                p_prob.append(df['tp_prob'][row][0])
                try:
                    p_res.append(df['P_residual'][row][0])
                except IndexError:
                    p_res.append(np.nan)
                    pass
            else:
                try:
                    p_res.append(df['P_residual'][row][1])
                except IndexError:
                    p_res.append(np.nan)
                    pass
                try:
                    p_pick.append(df['P_phasenet'][row][1])
                    p_prob.append(df['tp_prob'][row][1])
                except IndexError:
                    p_pick.append(np.nan)
                    p_prob.append(np.nan)
                    p_empty_count += 1
                    pass

            if not df['its'][row]:
                s_pick.append(np.nan)
                s_res.append(np.nan)
                s_prob.append(np.nan)
                s_empty_count += 1
            elif df['its'][row][0] != 1:
                s_pick.append(df['S_phasenet'][row][0])
                s_prob.append(df['ts_prob'][row][0])
                try:
                    s_res.append(df['S_residual'][row][0])
                except IndexError:
                    s_res.append(np.nan)
                    pass
            else:
                try:
                    s_res.append(df['S_residual'][row][1])
                except IndexError:
                    s_res.append(np.nan)
                    pass
                try:
                    s_pick.append(df['P_phasenet'][row][1])
                    s_prob.append(df['ts_prob'][row][1])
                except IndexError:
                    s_pick.append(np.nan)
                    s_prob.append(np.nan)
                    s_empty_count += 1
                    pass

    elif method == 'max_prob':
        for row in range(len(df)):
            fname.append(df['fname'][row])
            if not df['itp'][row]:
                p_pick.append(np.nan)
                p_res.append(np.nan)
                p_prob.append(np.nan)
                p_empty_count += 1
            else:
                p_pick.append(df['P_phasenet'][row][df['tp_prob'][row].index(max(df['tp_prob'][row]))])
                p_prob.append(max(df['tp_prob'][row]))
                try:
                    p_res.append(df['P_residual'][row][df['tp_prob'][row].index(max(df['tp_prob'][row]))])
                except IndexError:
                    p_res.append(np.nan)
            if not df['its'][row]:
                s_pick.append(np.nan)
                s_res.append(np.nan)
                s_prob.append(np.nan)
                s_empty_count += 1
            else:
                s_pick.append(df['S_phasenet'][row][df['ts_prob'][row].index(max(df['ts_prob'][row]))])
                s_prob.append(max(df['ts_prob'][row]))
                try:
                    s_res.append(df['S_residual'][row][df['ts_prob'][row].index(max(df['ts_prob'][row]))])
                except IndexError:
                    s_res.append(np.nan)
    else:
        print("Invalid method: method = (['earliest'], ['max_prob')]")

    pick_dict = {
        "P_phasenet": p_pick,
        "S_phasenet": s_pick,
        "P_res": p_res,
        "S_res": s_res,
        "P_prob": p_prob,
        "S_prob": s_prob,
        "pick_method": [method] * len(df),
        "fname": fname
    }

    df_picks = pd.DataFrame(pick_dict, columns=['P_phasenet', 'P_res', 'P_prob',
                                                'S_phasenet', 'S_res', 'S_prob', 'pick_method', 'fname'])

    df_picks = pd.merge(df_picks, df[['event_id', 'station', 'P_time', 'S_time', 'fname']], how='left', on='fname')
    df_picks = df_picks[
        ["event_id", "station", "P_time", "P_phasenet", "P_res", "P_prob", "S_time", "S_phasenet", "S_res", "S_prob",
         "pick_method", "fname"]].sort_values(['event_id', 'station'])
    print("Successfully isolated PhaseNet picks. Saving df_picks as new dataframe")
    return df_picks, p_empty_count, s_empty_count


df_picks, p_empty_count, s_empty_count = picker(df, PN_pick_method[0])

# def csvSync(dataset, output, arrival, sorted_headers, method):
#     log = pd.read_csv(os.path.join(dataset, 'data_log.csv'))
#     picks = pd.read_csv(os.path.join(output, 'picks.csv'))
#     arrivals = pd.read_pickle(arrival)
#
#     print("\nMerging picks.csv with data_log.csv...")
#     df = pd.merge(log, picks, how='left', on=['fname'])
#     print("Reformatting data from csv files...")
#     for col in ['start', 'end']:
#         df[col] = [obspy.UTCDateTime(x) for x in df[col]]
#
#     for col2 in ['itp', 'its']:
#         a = []
#         for x in range(len(df[col2])):
#             try:
#                 a.append(list(map(int, shlex.split(df[col2][x].strip('[]')))))
#             except AttributeError:
#                 a.append([])
#                 # print("Pick sample data is already in the correct format. Passing")
#                 pass
#         df[col2] = a
#
#     for col3 in ['tp_prob', 'ts_prob']:
#         b = []
#         for x in range(len(df[col3])):
#             try:
#                 b.append(list(map(float, shlex.split(df[col3][x].strip('[]')))))
#             except AttributeError:
#                 b.append([])
#                 # print("Pick probability data is already in the correct format. Passing")
#                 pass
#         df[col3] = b
#
#     utc_p_picks = []
#     utc_s_picks = []
#     for row in range(len(df['itp'])):
#         p_lst = df['itp'][row]
#         s_lst = df['its'][row]
#         p_lst2, s_lst2 = [], []
#         for p_element in p_lst:
#             p_lst2.append(df['start'][row] + float(p_element) * df['delta'][row])
#         for s_element in s_lst:
#             s_lst2.append(df['start'][row] + float(s_element) * df['delta'][row])
#         utc_p_picks.append(p_lst2)
#         utc_s_picks.append(s_lst2)
#
#     df['p_time'] = utc_p_picks
#     df['s_time'] = utc_s_picks
#
#     print("Merging with arrival.pickle...")
#     df = pd.merge(df, arrivals[["event_id", "station", "channel", "network", "P_time", "S_time"]],
#                   how='left', on=['event_id', 'station', 'network', 'channel'])
#
#     fname = []
#     p_picks = []
#     s_picks = []
#     p_diffs = []
#     p_diff = []
#     s_diffs = []
#     s_diff = []
#     p_prob_list = []
#     s_prob_list = []
#     p_empty_count = 0
#     s_empty_count = 0
#
#     for row2 in range(len(df['p_time'])):
#         fname.append(df['fname'][row2])
#         ptimes = df['p_time'][row2]
#         stimes = df['s_time'][row2]
#         pdiff_lst, sdiff_lst = [], []
#         for pt in ptimes:
#             try:
#                 pdiff_lst.append(df['P_time'][row2] - pt)
#             except TypeError:
#                 pass
#         p_diffs.append(pdiff_lst)
#         for st in stimes:
#             try:
#                 sdiff_lst.append(df['S_time'][row2] - st)
#             except TypeError:
#                 pass
#         s_diffs.append(sdiff_lst)
#         if method == 'earliest':
#             try:
#                 if df['itp'][row2][0] != 1:
#                     p_picks.append(df['p_time'][row2][0])
#                     p_diff.append(pdiff_lst[0])
#                     p_prob_list.append(df['tp_prob'][row2][0])
#                     # pick = 0
#                 else:
#                     p_picks.append(df['p_time'][row2][1])
#                     p_diff.append(pdiff_lst[1])
#                     p_prob_list.append(df['tp_prob'][row2][1])
#                     # pick = 1
#             except IndexError:
#                 pick = 2
#                 p_picks.append([])
#                 p_empty_count += 1
#                 p_diff.append(np.nan)
#                 p_prob_list.append(np.nan)
#
#             # if pick == 2:
#             #     p_picks.append([])
#             # else:
#             #     p_picks.append(df['p_time'][row2][pick])
#
#             try:
#                 if df['its'][row2][0] != 1:
#                     s_diff.append(sdiff_lst[0])
#                     s_prob_list.append(df['ts_prob'][row2][0])
#                     pick = 0
#                 else:
#                     s_diff.append(sdiff_lst[1])
#                     s_prob_list.append(df['ts_prob'][row2][1])
#                     pick = 1
#             except IndexError:
#                 pick = 2
#                 s_empty_count += 1
#                 s_diff.append(np.nan)
#                 s_prob_list.append(np.nan)
#
#             if pick == 2:
#                 s_picks.append(np.nan)
#             else:
#                 s_picks.append(df['s_time'][row2][pick])
#
#         elif method == 'max_prob':
#             try:
#                 p_diff.append(pdiff_lst[df['tp_prob'][row2].index(max(df['tp_prob'][row2]))])
#                 p_prob_list.append(max(df['tp_prob'][row2]))
#                 p_picks.append(df['p_time'][row2][df['tp_prob'][row2].index(max(df['tp_prob'][row2]))])
#             except ValueError:
#                 p_empty_count += 1
#                 p_diff.append(np.nan)
#                 p_prob_list.append(np.nan)
#                 p_picks.append(np.nan)
#             except IndexError:
#                 p_empty_count += 1
#                 p_diff.append(np.nan)
#                 p_prob_list.append(np.nan)
#                 p_picks.append(np.nan)
#             try:
#                 s_diff.append(sdiff_lst[df['ts_prob'][row2].index(max(df['ts_prob'][row2]))])
#                 s_prob_list.append(max(df['ts_prob'][row2]))
#                 s_picks.append(df['s_time'][row2][df['ts_prob'][row2].index(max(df['ts_prob'][row2]))])
#             except ValueError:
#                 s_empty_count += 1
#                 s_diff.append(np.nan)
#                 s_prob_list.append(np.nan)
#                 s_picks.append(np.nan)
#             except IndexError:
#                 s_empty_count += 1
#                 s_diff.append(np.nan)
#                 s_prob_list.append(np.nan)
#                 s_picks.append(np.nan)
#         else:
#             print("Invalid method: method = (['earliest'], ['max_prob')]")
#
#     df['P_residual'] = p_diffs
#     df['S_residual'] = s_diffs
#
#     df = df[sorted_headers].sort_values(['event_id', 'station'])
#     df.to_pickle(os.path.join(dataset, "data_log_merged.pickle"))
#     df.to_csv(os.path.join(dataset, "data_log_merged.csv"), index=False)
#     print("Merge successful. Copying files to ", dataset)
#
#     pick_dict = {
#         "P_phasenet": p_picks,
#         "S_phasenet": s_picks,
#         "P_res": p_diff,
#         "S_res": s_diff,
#         "P_prob": p_prob_list,
#         "S_prob": s_prob_list,
#         "pick_method": [method]*len(df),
#         "fname": fname
#     }
#
#     df_picks = pd.DataFrame(pick_dict, columns=['P_phasenet', 'P_res', 'P_prob',
#                                                 'S_phasenet', 'S_res', 'S_prob', 'pick_method', 'fname'])
#
#     df_picks = pd.merge(df_picks, df[['event_id', 'P_time', 'S_time', 'fname']], how='left', on='fname')
#     df_picks = df_picks[["event_id", "P_time", "P_phasenet", "P_res", "P_prob", "S_time", "S_phasenet", "S_res", "S_prob",
#                          "pick_method", "fname"]]
#
#     return df, df_picks, p_empty_count, s_empty_count
# df2, df_pick, p_empty, s_empty = csvSync(dataset_path, output_path, arrival_path, headers, PN_pick_method[0])

print("=================================================================================")


def outliers(df_picks, method, savepath):
    print("Initializing outlier detection algorithm... (method = {})".format(method))
    if method[0] == 'IQR':
        pq1, pq3 = np.nanpercentile(np.array(df_picks['P_res']), [25, 75])
        sq1, sq3 = np.nanpercentile(np.array(df_picks['S_res']), [25, 75])

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

    df_picks['P_inrange'] = df_picks['P_res'].between(p_lower_bound, p_upper_bound, inclusive=True)
    df_picks['S_inrange'] = df_picks['S_res'].between(s_lower_bound, s_upper_bound, inclusive=True)

    m = [method] * len(df_picks)
    df_picks['ol_method'] = m

    p_outliers = df_picks.loc[(df_picks['P_res'].notna()) & (df_picks['P_inrange'] == False)]
    p_outliers = p_outliers.drop(columns=["S_time", "S_phasenet", "S_res", "S_prob", "S_inrange"])
    p_outliers = p_outliers[["event_id", "P_time", "P_phasenet", "P_res", "P_prob", "P_inrange",
                             "pick_method", "ol_method", "fname"]]
    p_outliers.to_pickle(os.path.join(savepath, "p_outliers.pickle"))
    p_outliers.to_csv(os.path.join(savepath, "p_outliers.csv"), index=False)

    s_outliers = df_picks.loc[(df_picks['S_res'].notna()) & (df_picks['S_inrange'] == False)]
    s_outliers = s_outliers.drop(columns=["P_time", "P_phasenet", "P_res", "P_prob", "P_inrange"])
    s_outliers = s_outliers[["event_id", "S_time", "S_phasenet", "S_res", "S_prob", "S_inrange",
                             "pick_method", "ol_method", "fname"]]
    s_outliers.to_pickle(os.path.join(savepath, "s_outliers.pickle"))
    s_outliers.to_csv(os.path.join(savepath, "s_outliers.csv"), index=False)

    df_picks = df_picks[["event_id", "P_time", "P_phasenet", "P_res", "P_prob", "P_inrange",
                         "S_time", "S_phasenet", "S_res", "S_prob", "S_inrange",
                         "pick_method", "ol_method", "fname"]]

    df_picks.to_pickle(os.path.join(savepath, "filter_picks.pickle"))
    df_picks.to_csv(os.path.join(savepath, "filter_picks.csv"), index=False)
    print("Outliers isolated and saved to ", savepath)
    return df_picks, p_outliers, s_outliers


df_picks, p_out, s_out = outliers(df_picks, outlier_method, outlier_path)
print("=================================================================================")


def Histogram(df_picks, phase, p_out, s_out, plot_path, Methods, method):
    fig, ax = plt.subplots()
    if phase == 'P':
        c = '#1f77b4'
        d = df_picks['P_res'][df_picks['P_inrange']]
        ol = len(p_out)
        path = os.path.join(plot_path, 'p_hist.png')
    elif phase == 'S':
        c = '#ff7f0e'
        d = df_picks['S_res'][df_picks['S_inrange']]
        ol = len(s_out)
        path = os.path.join(plot_path, 's_hist.png')
    else:
        print("Invalid phase entry: phase= ('P', 'S')")

    n, bins, patches = plt.hist(x=d, bins='auto', color=c,
                                alpha=0.7, rwidth=0.85)

    plt.xlim(-method[1], method[1])

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Residual time (s)')
    plt.ylabel('Frequency*')
    plt.title('Histogram of {}-pick Residuals (expected-observed)'.format(phase))
    textstr = '\n'.join((
        r'$n=%.0f$' % (len(d),),
        r'$\mu=%.4f$' % (statistics.mean(d),),
        r'$\mathrm{median}=%.4f$' % (round(statistics.median(d), 4),),
        r'$\sigma=%.4f$' % (statistics.pstdev(d),)))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='square,pad=.6', facecolor='lightgrey', edgecolor='black', alpha=1))
    plt.annotate('*method={}: {} '.format(method, Methods[method[0]]), (0, 0), (0, -40),
                 xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
    plt.annotate('{} total arrivals removed, {} outliers removed'.format(len(df_picks) - len(d), ol), (0, 0), (0, -50),
                 xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(path)
    plt.show()


Histogram(df_picks, 'P', p_out, s_out, plot_path, methods, outlier_method)
Histogram(df_picks, 'S', p_out, s_out, plot_path, methods, outlier_method)


# %%
def scatterPlot(df_picks, plot_path, method):
    p_res_np = abs(df_picks['P_res'][df_picks['P_inrange']])
    s_res_np = abs(df_picks['S_res'][df_picks['S_inrange']])
    p_prob_np = df_picks['P_prob'][df_picks['P_inrange']]
    s_prob_np = df_picks['S_prob'][df_picks['S_inrange']]

    px = np.linspace(0, method[1], len(p_res_np))
    sx = np.linspace(0, method[1], len(s_res_np))
    p_slope, p_intercept, pr_value, pp_value, p_std_err = stats.linregress(p_res_np, p_prob_np)
    s_slope, s_intercept, sr_value, sp_value, s_std_err = stats.linregress(s_res_np, s_prob_np)

    plt.close()
    fig = plt.figure(constrained_layout=False)
    fig.suptitle('Scatterplot of P & S Residuals by PhaseNet Probability')

    gs1 = fig.add_gridspec(ncols=1, nrows=1, top=0.93, bottom=0.3, left=0.1, right=0.95)
    gs2 = fig.add_gridspec(ncols=1, nrows=2, top=0.25, bottom=0.05, left=0.1, right=0.95, hspace=0.02)

    ax1 = plt.subplot(gs1[0], xlim=(0, method[1]))
    ax1.grid(axis='both', alpha=0.4)
    ax1.set(ylabel='PhaseNet probability*')
    pl, = ax1.plot(px, p_intercept + p_slope * px, color='#333333', linestyle='dashdot',lw = 2.4,
                   label="y=%.2fx+%.2f" % (p_slope, p_intercept))
    sl, = ax1.plot(sx, s_intercept + s_slope * sx, color='#333333', linestyle='dotted',lw=2.65,
                   label="y=%.2fx+%.2f" % (s_slope, s_intercept))
    pscat = ax1.scatter(p_res_np, p_prob_np)
    pscat.set_label('P')
    sscat = ax1.scatter(s_res_np, s_prob_np)
    sscat.set_label('S')
    leg = ax1.legend([pscat, pl, sscat, sl],
                     ['P', "y=%.2fx+%.2f" % (p_slope, p_intercept), 'S', "y=%.2fx+%.2f" % (s_slope, s_intercept)])
    leg.get_frame().set_edgecolor('#262626')

    ax2 = plt.subplot(gs2[0], frame_on=False, xlim=(0, method[1]))
    ax2.tick_params(axis=u'both', which=u'both', length=0)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.boxplot(p_res_np, vert=False, whis=(0, 100), patch_artist=True,
                boxprops=dict(facecolor='#1f77b4', color='k'),
                medianprops=dict(color='k'),
                )

    ax3 = plt.subplot(gs2[1], frame_on=False, xlim=(0, method[1]), xlabel='abs(Residual time) (s)')
    ax3.tick_params(axis=u'both', which=u'both', length=0)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.boxplot(s_res_np, vert=False, whis=(0, 100), patch_artist=True,
                boxprops=dict(facecolor='#ff7f0e', color='k'),
                medianprops=dict(color='k'),
                )

    # fc = colors.to_rgba('lightgrey')
    # ec = colors.to_rgba('black')
    # fc = fc[:-1] + (0.7,)
    # plt.annotate("y=%.3fx+%.3f" % (p_slope, p_intercept), xy=(.715, .99), xytext=(12, -12), va='top',
    #              xycoords='axes fraction', textcoords='offset points',
    #              bbox=dict(facecolor=fc, edgecolor=ec, boxstyle='square,pad=.6'))

    # plt.annotate('*method={}: {} '.format(method, Methods[method[0]]), (0, 0), (0, -40),
    #              xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
    # plt.annotate('{} total arrivals removed, {} outliers removed'
    #              .format(2 * len(df_picks) - len(p_res_np) - len(s_res_np), len(p_out) + len(s_out)), (0, 0), (0, -50),
    #              xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)

    plt.savefig(os.path.join(plot_path, 'scatter.png'))

    print("P:   y=%.6fx+(%.6f)" % (p_slope, p_intercept))
    print("S:   y=%.6fx+(%.6f)" % (s_slope, s_intercept))
    plt.show()
    plt.close()


scatterPlot(df_picks, plot_path, outlier_method)
# %%

print("{} P outliers excluded".format(len(p_out)))
print("{} S outliers excluded".format(len(s_out)))

pssr = np.nansum([i ** 2 for i in df_picks['P_res']])
sssr = np.nansum([i ** 2 for i in df_picks['S_res']])
print("P-SSR = ", pssr)
print("S-SSR - ", sssr)
