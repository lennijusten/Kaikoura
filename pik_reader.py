# PickVisualization
# Author: Lennart Justen
# Last revision: 6/13/20

# Description:

import datetime
import obspy
import numpy as np
import os
import pandas as pd
import shlex
import matplotlib.pyplot as plt
from scipy import stats

PN_pick_method = ['min_res']
outlier_method = ['over', 2]
vps_method = ['outlier']

tbegin = -30  # starttime is 30 seconds prior to origin of earthquake
tend = 100  # end time is 240 seconds after origin of earthquake
dt = 0.01
n_samp = int((tend - tbegin) / dt + 1)

tmin = 0
tmax = 100

p_threshold = 0.05
s_threshold = 0.05

dataset_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset'
outlier_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset/Outliers'
output_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/PhaseNet/output'
arrival_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/Arrival.pickle'
event_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/Event.pickle'
station_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/Station.pickle'
plot_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset/Plots'
sac_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/'

# METHODS:
# earliest -- Uses PhaseNet's earliest pick as the arrival pick
# max_prob -- Uses the PhaseNet pick with the highest probability as the arrival pick
# hybrid -- Prioritize early picks unless there is a high probability

methods = {
    "earliest": 'Uses PhaseNets earliest pick as the arrival pick.',
    "max_prob": 'Uses the PhaseNet pick with the highest probability as the arrival pick.',
    "min_res": 'PhaseNet picks with the smallest residual (cheating method)',
    "IQR": 'IQR outliers excluded',
    "over": 'outliers over a limit excluded',
    "range": 'vps ratios in range'
}


def checkTimeInt(tbegin, tend, tmin, tmax):
    if tmin < tbegin:
        tmin = tbegin
        print("WARNING: tmin is set before queried start time. Setting to tbegin")
    if tmax > tend:
        tmax = tend
        print("WARNING: tmax is set after queried end time. Setting to tend")
    return tmin, tmax


tmin, tmax = checkTimeInt(tbegin, tend, tmin, tmax)


def initFrames(dataset_path, output_path, arrival_path, event_path, station_path):
    print("Initializing dataframes...")
    log = pd.read_csv(os.path.join(dataset_path, 'data_log.csv'))
    picks = pd.read_csv(os.path.join(output_path, 'picks.csv'))
    arrivals = pd.read_pickle(arrival_path)
    events = pd.read_pickle(event_path)
    stations = pd.read_pickle(station_path)
    return log, picks, arrivals, events, stations


log, picks, arrivals, events, stations = initFrames(dataset_path, output_path, arrival_path, event_path, station_path)


def pickConverter(picks):
    print("Cleaning PhaseNet pick data...")
    for col2 in ['itp', 'its']:
        a = []
        for x in range(len(picks)):
            try:
                a.append(list(map(int, shlex.split(picks[col2][x].strip('[]')))))
            except AttributeError:
                a.append([])
                pass
        picks[col2] = a

    for col3 in ['tp_prob', 'ts_prob']:
        b = []
        for x in range(len(picks)):
            try:
                b.append(list(map(float, shlex.split(picks[col3][x].strip('[]')))))
            except AttributeError:
                b.append([])
                pass
        picks[col3] = b
    return picks


picks = pickConverter(picks)


def probThresholder(picks, p_thresh, s_thresh):
    print("Removing picks below probability thresholds...")
    for row in range(len(picks)):
        p_idx = [i for i in range(len(picks['tp_prob'][row])) if picks['tp_prob'][row][i] < p_thresh]
        for index in sorted(p_idx, reverse=True):
            del picks['tp_prob'][row][index]
            del picks['itp'][row][index]

        s_idx = [i for i in range(len(picks['ts_prob'][row])) if picks['ts_prob'][row][i] < s_thresh]
        for index in sorted(s_idx, reverse=True):
            del picks['ts_prob'][row][index]
            del picks['its'][row][index]
    return picks


picks = probThresholder(picks, p_threshold, s_threshold)


def timeConverter(df):
    print("Converting arrivals to UTC DateTime...")
    for col in ['start', 'end']:
        df[col] = [obspy.UTCDateTime(x) for x in df[col]]
    return df


log = timeConverter(log)

df = pd.merge(log, picks, how='left', on=['fname'])
df = pd.merge(df, arrivals[["event_id", "station", "channel", "network", "P_time", "S_time"]],
              how='left', on=['event_id', 'station', 'network', 'channel'])


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


def dropNAs(df):
    print("Dropping arrivals that do not include a P & S GeoNet time...")
    df = df.drop(df.loc[(df['P_time'].isna() == True) & (df['S_time'].isna() == True)].index)
    return df


df = dropNAs(df)


def timeThresholder(df, tmin, tmax):
    print("Dropping arrivals outside of time window...")
    init_len = len(df)
    p_delay = df.loc[(df['P_time']-df['start'] >= tmax) | (df['P_time']-df['start'] <= tmin)]
    s_delay = df.loc[(df['S_time']-df['start'] >= tmax) | (df['S_time']-df['start'] <= tmin)]

    out_window = pd.concat([p_delay, s_delay], axis=1, sort=False, join='outer')

    df = df[~df.index.isin(out_window.index)]
    df = df.reset_index()
    final_len = len(df)
    print("{} rows dropped (tmin = {}, tmax = {})".format(init_len-final_len, tmin, tmax))

    df['tmin'] = [tmin]*len(df)
    df['tmax'] = [tmax]*len(df)
    return df


df = timeThresholder(df, tmin, tmax)


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

headers = ['network', 'event_id', 'station', 'channel', 'samples', 'delta', 'start', 'end', 'P_residual', 'P_time',
           'P_phasenet', 'tp_prob', 'itp', 'S_residual', 'S_time', 'S_phasenet', 'ts_prob', 'its',
           'tmin', 'tmax', 'filter_method', 'fname']
df = df[headers].sort_values(['event_id', 'station'])
df.to_pickle(os.path.join(dataset_path, "data_log_merged.pickle"))
df.to_csv(os.path.join(dataset_path, "data_log_merged.csv"), index=False)
print("Success! Merged dataframe being saved to ", dataset_path)

print("=================================================================================")


def picker(df, p_thresh, s_thresh, method, savepath):
    print("Initializing pick algorithm... (method = {})".format(method))
    fname = []
    vps = []

    p_pick = []
    itp = []
    p_res = []
    p_prob = []
    p_empty_count = 0

    s_pick = []
    its = []
    s_res = []
    s_prob = []
    s_empty_count = 0

    if method == 'earliest':
        for row in range(len(df)):
            fname.append(df['fname'][row])
            if not df['itp'][row]:
                p_pick.append(np.nan)
                itp.append(np.nan)
                p_res.append(np.nan)
                p_prob.append(np.nan)
                p_empty_count += 1
            elif df['itp'][row][0] != 1:
                p_pick.append(df['P_phasenet'][row][0])
                itp.append(int(df['itp'][row][0]))
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
                    itp.append(int(df['itp'][row][1]))
                    p_prob.append(df['tp_prob'][row][1])
                except IndexError:
                    p_pick.append(np.nan)
                    itp.append(np.nan)
                    p_prob.append(np.nan)
                    p_empty_count += 1
                    pass

            if not df['its'][row]:
                s_pick.append(np.nan)
                its.append(np.nan)
                s_res.append(np.nan)
                s_prob.append(np.nan)
                s_empty_count += 1
            elif df['its'][row][0] != 1:
                s_pick.append(df['S_phasenet'][row][0])
                its.append(int(df['its'][row][0]))
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
                    s_pick.append(df['S_phasenet'][row][1])
                    its.append(int(df['its'][row][1]))
                    s_prob.append(df['ts_prob'][row][1])
                except IndexError:
                    s_pick.append(np.nan)
                    its.append(np.nan)
                    s_prob.append(np.nan)
                    s_empty_count += 1
                    pass
    elif method == 'max_prob':
        for row in range(len(df)):
            fname.append(df['fname'][row])
            if not df['itp'][row]:
                p_pick.append(np.nan)
                itp.append(np.nan)
                p_res.append(np.nan)
                p_prob.append(np.nan)
                p_empty_count += 1
            else:
                if int(df['itp'][row][df['tp_prob'][row].index(max(df['tp_prob'][row]))]) != 1:
                    p_pick.append(df['P_phasenet'][row][df['tp_prob'][row].index(max(df['tp_prob'][row]))])
                    itp.append(int(df['itp'][row][df['tp_prob'][row].index(max(df['tp_prob'][row]))]))
                    p_prob.append(max(df['tp_prob'][row]))
                    try:
                        p_res.append(df['P_residual'][row][df['tp_prob'][row].index(max(df['tp_prob'][row]))])
                    except IndexError:
                        p_res.append(np.nan)
                else:
                    try:
                        p_res.append(df['P_residual'][row][1])
                    except IndexError:
                        p_res.append(np.nan)
                        pass
                    try:
                        p_pick.append(df['P_phasenet'][row][1])
                        itp.append(int(df['itp'][row][1]))
                        p_prob.append(df['tp_prob'][row][1])
                    except IndexError:
                        p_pick.append(np.nan)
                        itp.append(np.nan)
                        p_prob.append(np.nan)
                        p_empty_count += 1
                        pass
            if not df['its'][row]:
                s_pick.append(np.nan)
                its.append(np.nan)
                s_res.append(np.nan)
                s_prob.append(np.nan)
                s_empty_count += 1
            else:
                if int(df['its'][row][df['ts_prob'][row].index(max(df['ts_prob'][row]))]) != 1:
                    s_pick.append(df['S_phasenet'][row][df['ts_prob'][row].index(max(df['ts_prob'][row]))])
                    its.append(int(df['its'][row][df['ts_prob'][row].index(max(df['ts_prob'][row]))]))
                    s_prob.append(max(df['ts_prob'][row]))
                    try:
                        s_res.append(df['S_residual'][row][df['ts_prob'][row].index(max(df['ts_prob'][row]))])
                    except IndexError:
                        s_res.append(np.nan)
                else:
                    try:
                        s_res.append(df['S_residual'][row][1])
                    except IndexError:
                        s_res.append(np.nan)
                        pass
                    try:
                        s_pick.append(df['S_phasenet'][row][1])
                        its.append(int(df['its'][row][1]))
                        s_prob.append(df['ts_prob'][row][1])
                    except IndexError:
                        s_pick.append(np.nan)
                        its.append(np.nan)
                        s_prob.append(np.nan)
                        s_empty_count += 1
                        pass
    elif method == 'hybrid':
        pass
    elif method == 'min_res':
        for row in range(len(df)):
            fname.append(df['fname'][row])
            if not df['P_residual'][row]:
                p_pick.append(np.nan)
                p_res.append(np.nan)
                p_prob.append(np.nan)
                itp.append(np.nan)
                p_empty_count += 1
            else:
                p_pick.append(df['P_phasenet'][row][(np.abs(df['P_residual'][row])).argmin()])
                p_prob.append(df['tp_prob'][row][(np.abs(df['P_residual'][row])).argmin()])
                p_res.append(df['P_residual'][row][(np.abs(df['P_residual'][row])).argmin()])
                itp.append(df['itp'][row][(np.abs(df['P_residual'][row])).argmin()])
            if not df['S_residual'][row]:
                s_pick.append(np.nan)
                s_res.append(np.nan)
                s_prob.append(np.nan)
                its.append(np.nan)
                s_empty_count += 1
            else:
                s_pick.append(df['S_phasenet'][row][(np.abs(df['S_residual'][row])).argmin()])
                s_prob.append(df['ts_prob'][row][(np.abs(df['S_residual'][row])).argmin()])
                s_res.append(df['S_residual'][row][(np.abs(df['S_residual'][row])).argmin()])
                its.append(df['its'][row][(np.abs(df['S_residual'][row])).argmin()])
    else:
        print("Invalid method: method = (['earliest'], ['max_prob')]")

    for row in range(len(df)):
        try:
            vps.append(its[row] / itp[row])
        except ValueError:
            vps.append(np.nan)
            pass

    pick_dict = {
        "P_phasenet": p_pick,
        "S_phasenet": s_pick,
        "P_res": p_res,
        "S_res": s_res,
        "P_prob": p_prob,
        "S_prob": s_prob,
        "itp": itp,
        "its": its,
        "vps": vps,
        "P_thresh": [p_thresh] * len(df),
        "S_thresh": [s_thresh] * len(df),
        "pick_method": [method] * len(df),
        "fname": fname
    }

    df_picks = pd.DataFrame(pick_dict, columns=['P_phasenet', 'P_res', 'P_prob', 'itp', 'P_thresh',
                                                'S_phasenet', 'S_res', 'S_prob', 'its', 'S_thresh',
                                                'vps', 'pick_method', 'fname'])
    # df_picks['itp'] = df_picks['itp'].astype('Int64')
    # df_picks['its'] = df_picks['its'].astype('Int64')

    df_picks = pd.merge(df_picks, df[['event_id', 'network', 'station', 'P_time', 'S_time',
                                      'tmin', 'tmax', 'filter_method', 'fname']], how='left', on='fname')
    df_picks = df_picks[
        ["event_id", "network", "station", "P_time", "P_phasenet", "P_res", "P_prob", "itp", "P_thresh",
         "S_time", "S_phasenet", "S_res", "S_prob", "its", "S_thresh",
         "vps", "pick_method", "tmin", "tmax", "filter_method", "fname"]].sort_values(['event_id', 'station'])

    df_picks.to_pickle(os.path.join(savepath, "filter_picks.pickle"))
    df_picks.to_csv(os.path.join(savepath, "filter_picks.csv"), index=False)

    print("Successfully isolated PhaseNet picks. Saving df_picks as new dataframe")
    return df_picks, p_empty_count, s_empty_count


df_picks, p_empty_count, s_empty_count = picker(df, p_threshold, s_threshold, PN_pick_method[0], outlier_path)

print("=================================================================================")


def outliers(df_picks, method, savepath):
    print("Initializing residual outlier detection algorithm... (method = {})".format(method))
    m = [method] * len(df_picks)
    df_picks['ol_method'] = m

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
        print("Invalid residual outlier method: method = (['IQR'], ['over', limit])")

    print("P Lower bound ({} method): {}".format(method[0], p_lower_bound))
    print("P Upper bound ({} method): {}".format(method[0], p_upper_bound))
    print("S Lower bound ({} method): {}".format(method[0], s_lower_bound))
    print("S Upper bound ({} method): {}".format(method[0], s_upper_bound))

    df_picks['P_inrange'] = df_picks['P_res'].between(p_lower_bound, p_upper_bound, inclusive=True)
    df_picks['S_inrange'] = df_picks['S_res'].between(s_lower_bound, s_upper_bound, inclusive=True)

    p_outliers = df_picks.loc[(df_picks['P_res'].notna()) & (df_picks['P_inrange'] == False)]
    s_outliers = df_picks.loc[(df_picks['S_res'].notna()) & (df_picks['S_inrange'] == False)]

    p_outliers = p_outliers[["event_id", "station", "P_time", "P_phasenet", "P_res", "P_prob", "itp", "P_inrange",
                             "pick_method", "ol_method", "filter_method", "fname"]]
    s_outliers = s_outliers[["event_id", "station", "S_time", "S_phasenet", "S_res", "S_prob", "its", "S_inrange",
                             "pick_method", "ol_method", "filter_method", "fname"]]

    p_outliers.to_pickle(os.path.join(savepath, "p_outliers.pickle"))
    p_outliers.to_csv(os.path.join(savepath, "p_outliers.csv"), index=False)
    s_outliers.to_pickle(os.path.join(savepath, "s_outliers.pickle"))
    s_outliers.to_csv(os.path.join(savepath, "s_outliers.csv"), index=False)

    df_picks = df_picks[
        ["event_id", "network", "station", "P_time", "P_phasenet", "P_res", "P_prob", "itp", "P_inrange", "P_thresh",
         "S_time", "S_phasenet", "S_res", "S_prob", "its", "S_inrange", "S_thresh",
         "vps", "pick_method", "ol_method", "filter_method", "tmin", "tmax", "fname"]]

    df_picks.to_pickle(os.path.join(savepath, "filter_picks.pickle"))
    df_picks.to_csv(os.path.join(savepath, "filter_picks.csv"), index=False)
    print("Residual outliers isolated and saved to ", savepath)
    return df_picks, p_outliers, s_outliers


df_picks, p_out, s_out = outliers(df_picks, outlier_method, outlier_path)


def vpsOutliers(df_picks, p_out, s_out, method, savepath):
    print("Initializing VPS outlier detection algorithm... (method = {})".format(method))
    m = [method] * len(df_picks)
    df_picks['vps_ol_method'] = m

    if method[0] == 'IQR':
        vpsq1, vpsq3 = np.nanpercentile(np.array(df_picks['vps']), [25, 75])
        vps_iqr = vpsq3 - vpsq1
        vps_lower_bound = vpsq1 - (1.5 * vps_iqr)
        vps_upper_bound = vpsq3 + (1.5 * vps_iqr)

        df_picks['vps_inrange'] = df_picks['vps'].between(vps_lower_bound, vps_upper_bound, inclusive=True)
    elif method[0] == 'range':
        vps_lower_bound = method[1]
        vps_upper_bound = method[2]

        df_picks['vps_inrange'] = df_picks['vps'].between(vps_lower_bound, vps_upper_bound, inclusive=True)
    elif method[0] == 'outlier':
        fn = pd.concat([p_out['fname'], s_out['fname']])
        df_picks['vps_inrange'] = (df_picks['fname'].isin(fn) == False) & (df_picks['vps'].notna())
    else:
        print("Invalid VPS outlier method: method = (['IQR'], ['range', lower, upper])")

    vps_outliers = df_picks.loc[(df_picks['vps'].notna()) & (df_picks['vps_inrange'] == False)]
    vps_outliers = vps_outliers[["event_id", "station", "P_time", "P_phasenet", "P_res", "P_prob", "itp", "P_inrange",
                                 "S_time", "S_phasenet", "S_res", "S_prob", "its", "S_inrange", "vps", "vps_inrange",
                                 "pick_method", "ol_method", "filter_method", "fname"]]

    vps_outliers.to_pickle(os.path.join(savepath, "vps_outliers.pickle"))
    vps_outliers.to_csv(os.path.join(savepath, "vps_outliers.csv"), index=False)

    df_picks = df_picks[
        ["event_id", "network", "station", "P_time", "P_phasenet", "P_res", "P_prob", "itp", "P_inrange", "P_thresh",
         "S_time", "S_phasenet", "S_res", "S_prob", "its", "S_inrange", "S_thresh", "vps", "vps_inrange",
         "pick_method", "ol_method", "vps_ol_method", "tmin", "tmax", "filter_method", "fname"]]

    print("VPS outliers isolated and saved to ", savepath)
    return df_picks, vps_outliers


df_picks, vps_out = vpsOutliers(df_picks, p_out, s_out, vps_method, outlier_path)

print("=================================================================================")


def eventParser(df_picks, events, p_out, s_out, savepath):
    print("Parsing arrivals by event_id...")

    # Event statistics
    event_list = []
    event_rating = []
    rms = []
    n = []
    prob_mean = []
    prob_std = []

    p_rms = []
    N_p = []
    Np_out = []
    p_mean_res = []
    p_median_res = []
    p_prob_mean = []
    p_prob_std = []

    s_rms = []
    N_s = []
    Ns_out = []
    s_mean_res = []
    s_median_res = []
    s_prob_mean = []
    s_prob_std = []

    for event in df_picks['event_id'].unique():
        event_list.append(event)
        df_temp = df_picks.loc[df_picks['event_id'] == event]

        pssr = np.nansum([i ** 2 for i in df_temp['P_res'][df_temp['P_inrange']]])
        sssr = np.nansum([i ** 2 for i in df_temp['S_res'][df_temp['S_inrange']]])

        Np = df_temp['P_res'][df_temp['P_inrange']].count()
        Ns = df_temp['S_res'][df_temp['S_inrange']].count()

        try:
            prms = np.sqrt(pssr / (Np - 1))
        except RuntimeWarning:
            prms = np.nan
        try:
            srms = np.sqrt(sssr / (Ns - 1))
        except RuntimeWarning:
            srms = np.nan

        try:
            RMS = np.sqrt((pssr + sssr) / (Np + Ns - 1))
        except RuntimeWarning:
            RMS = np.nan

        event_rating.append(np.nan)
        rms.append(RMS)
        n.append(len(df_temp))
        prob_mean.append(np.mean(pd.concat([df_temp['P_prob'][df_temp['P_inrange']],
                                            df_temp['S_prob'][df_temp['S_inrange']]])))
        prob_std.append(np.std(pd.concat([df_temp['P_prob'][df_temp['P_inrange']],
                                          df_temp['S_prob'][df_temp['S_inrange']]])))

        try:
            p_rms.append(prms)
            N_p.append(Np)
            Np_out.append(len(p_out.loc[p_out['event_id'] == event]))
            p_mean_res.append(np.nanmean(df_temp['P_res'][df_temp['P_inrange']]))
            p_median_res.append(np.nanmedian(df_temp['P_res'][df_temp['P_inrange']]))
            p_prob_mean.append(np.nanmean(df_temp['P_prob'][df_temp['P_inrange']]))
            p_prob_std.append(np.std(df_temp['P_prob'][df_temp['P_inrange']]))

            s_rms.append(srms)
            N_s.append(Ns)
            Ns_out.append(len(s_out.loc[s_out['event_id'] == event]))
            s_mean_res.append(np.nanmean(df_temp['S_res'][df_temp['S_inrange']]))
            s_median_res.append(np.nanmedian(df_temp['S_res'][df_temp['S_inrange']]))
            s_prob_mean.append(np.nanmean(df_temp['S_prob'][df_temp['S_inrange']]))
            s_prob_std.append(np.std(df_temp['S_prob'][df_temp['S_inrange']]))
        except RuntimeWarning:
            pass

    event_dict = {
        "event_id": event_list,
        "event_rating": event_rating,
        "n_arrivals": n,
        "rms": rms,
        "P_res": p_rms,
        "S_rms": s_rms,
        "nP": N_p,
        "nS": N_s,
        "nP_outliers": Np_out,
        "nS_outliers": Ns_out,
        "P_mean_res": p_mean_res,
        "S_mean_res": s_mean_res,
        "P_median_res": p_median_res,
        "S_median_res": s_median_res,
        "prob_mean": prob_mean,
        "prob_std": prob_std,
        "P_prob_mean": p_prob_mean,
        "S_prob_mean": s_prob_mean,
        "P_prob_std": p_prob_std,
        "S_prob_std": s_prob_std,
        "pick_method": [df_temp['pick_method'].iloc[0]] * len(df_picks['event_id'].unique()),
        "ol_method": [df_temp['ol_method'].iloc[0]] * len(df_picks['event_id'].unique()),
        "filter_method": [df_temp['filter_method'].iloc[0]] * len(df_picks['event_id'].unique())
    }

    df_events = pd.DataFrame.from_dict(event_dict)
    df_events = pd.merge(df_events, events, on='event_id', how='left')

    df_events.to_pickle(os.path.join(savepath, "event_comparison.pickle"))
    df_events.to_csv(os.path.join(savepath, "event_comparison.csv"), index=False)
    print("Saving event comparison to ", savepath)
    return df_events


df_events = eventParser(df_picks, events, p_out, s_out, dataset_path)


# %%
def eventPlotter(df_events, x, x_descrip, min_arrivals, option, savepath):
    df_events_filt = df_events.loc[df_events['n_arrivals'] >= min_arrivals]

    n, bins, patches = plt.hist(x=df_events_filt['rms'], bins=30, facecolor='#008080', alpha=0.7, rwidth=0.85)
    mu = np.mean(df_events_filt['rms'])
    sigma = np.std(df_events_filt['rms'])

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('P rms')
    plt.ylabel('Frequency')
    plt.title('Histogram of rms by events')

    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(savepath)
    plt.show()

    fig = plt.figure()
    fig.suptitle('RMS by {}'.format(x_descrip))

    xlin = np.linspace(np.min(df_events_filt[x]), np.max(df_events_filt[x]), len(df_events_filt[x]))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_events_filt[x], df_events_filt['rms'])
    if option == 'boxplot':
        gs1 = fig.add_gridspec(ncols=1, nrows=1, top=0.93, bottom=0.2, left=0.13, right=0.95)
        gs2 = fig.add_gridspec(ncols=1, nrows=1, top=0.15, bottom=0.02, left=0.1, right=0.95)

        ax1 = plt.subplot(gs1[0], xlim=(0, 1))
        ax1.grid(axis='both', alpha=0.4)
        ax1.set(ylabel='RMS', xlabel=x_descrip)

        ax1.scatter(df_events_filt[x], df_events_filt['rms'], c='#008080')
        fit_l, = ax1.plot(xlin, intercept + slope * xlin, color='#333333', linestyle=':', lw=2.4)

        leg = ax1.legend([fit_l],
                         ["linear fit: y=%.2fx+%.0f" % (slope, intercept)],
                         loc='upper left')
        leg.get_frame().set_edgecolor('#262626')

        ax2 = plt.subplot(gs2[0], frame_on=False)
        ax2.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.boxplot(df_events_filt[x], vert=False, whis=(0, 100), patch_artist=True,
                    boxprops=dict(facecolor='#008080', color='k'),
                    medianprops=dict(color='k'),
                    )
    elif option == 'PDF':
        gs1 = fig.add_gridspec(ncols=1, nrows=1, top=0.93, bottom=0.23, left=0.1, right=0.95)
        gs2 = fig.add_gridspec(ncols=1, nrows=1, top=0.13, bottom=0.035, left=0.1, right=0.95)

        ax1 = plt.subplot(gs1[0])
        ax1.grid(axis='both', alpha=0.4)
        ax1.set(ylabel='RMS', xlabel=x_descrip)

        ax1.scatter(df_events_filt[x], df_events_filt['rms'], c='#008080')
        fit_l, = ax1.plot(xlin, intercept + slope * xlin, color='#333333', linestyle=':', lw=2.4)

        leg = ax1.legend([fit_l],
                         ["linear fit: y=%.2fx+%.0f" % (slope, intercept)],
                         loc='upper left')
        leg.get_frame().set_edgecolor('#262626')

        ax2 = plt.subplot(gs2[0], frame_on=True)
        n, bins, patches = plt.hist(df_events_filt[x], bins='auto')
        y = stats.norm.pdf(bins, np.mean(df_events_filt[x]), np.std(df_events_filt[x]))

        ax2 = plt.subplot(gs2[0])
        # ax2.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)
        # plt.setp(ax2.get_yticklabels(), visible=False)

        ax2.plot(bins, y, '--', c='#008080', linewidth=2)
        plt.ylabel("PDF")

    else:
        print("Invalid option arg: ('boxplot', 'PDF')")

    plt.savefig(os.path.join(plot_path, 'wadati.png'))
    plt.show()
    plt.close()


eventPlotter(df_events, 'prob_mean', 'Prob mean', 20, 'PDF', plot_path)

# %%
def sort_by_distance(st):
    # grab values to sort
    sortvals = []
    for tr in st:
        sortvals.append(tr.stats['distance'])

    # grab indices after sorting
    isort = np.argsort(sortvals)

    # resort stream
    sortedstream = st.sort(['distance'])

    return sortedstream


def multiEventPlotter(df_events, stations, sac_path, scale, clip, savepath):
    for event in df_events['event_id']:
        st = obspy.read(os.path.join(sac_path, event, '*.SAC'))

        # Filter data
        st = st.detrend()
        st = st.filter('bandpass', freqmin=1, freqmax=15)

        # Calculate distance
        ievt = df_events['event_id'].values == event
        elat = df_events['event_latitude'].values[ievt]
        elon = df_events['event_longitude'].values[ievt]
        for tr in st:
            # Get station location
            ista = stations['station_name'].values == tr.stats.station
            slat = stations['station_latitude'].values[ista]
            slon = stations['station_longitude'].values[ista]

            tr.stats.distance = obspy.geodetics.gps2dist_azimuth(slat, slon, elat, elon)[0]

        st = sort_by_distance(st)

        # Normalize traces by max amplitude, scale and clip data
        delta = 0
        for tr in st:
            tr.data = tr.data / max(abs(tr.data)) * scale
            tr.data[abs(tr.data) > clip] = np.sign(tr.data[abs(tr.data) > clip]) * clip

            if tr.stats.channel[-1] == 'E':
                color = 'green'
            elif tr.stats.channel[-1] == 'N':
                color = 'blue'
            elif tr.stats.channel[-1] == 'Z':
                color = 'black'

                plt.plot(tr.times(), tr.data + delta, color=color)
                plt.fill_between(tr.times(), tr.data + delta, delta, where=(tr.data + delta > delta), color='black')
                plt.text(0, delta, tr.stats.station + ' ' + tr.stats.channel)
                plt.savefig(os.path.join(savepath, event))
                # plt.show()
                delta += 1
            else:
                color = 'purple'

        plt.show()

# multiEventPlotter(df_events, stations, sac_path, 4, 0.75, "/Volumes/WMEL/Kaikoura/Event Plots")

def getDistance(df_picks, df_events, stations, sac_path):
    distance = []
    for (net, event, sta) in (zip(df_picks['network'], df_picks['event_id'], df_picks['station'])):
        st = obspy.read(os.path.join(sac_path, event, net+'_'+sta+'*.SAC'))

        ievt = df_events['event_id'].values == event
        elat = df_events['event_latitude'].values[ievt]
        elon = df_events['event_longitude'].values[ievt]

        ista = stations['station_name'].values == st[0].stats.station
        slat = stations['station_latitude'].values[ista]
        slon = stations['station_longitude'].values[ista]

        distance.append(obspy.geodetics.gps2dist_azimuth(slat, slon, elat, elon)[0])

    df_picks['distance'] = [idx/1000 for idx in distance]
    df_picks = df_picks[
        ["event_id", "network", "station", "distance", "P_time", "P_phasenet", "P_res", "P_prob", "itp", "P_inrange", "P_thresh",
         "S_time", "S_phasenet", "S_res", "S_prob", "its", "S_inrange", "S_thresh", "vps", "vps_inrange",
         "pick_method", "ol_method", "vps_ol_method", "tmin", "tmax", "filter_method", "fname"]]

    return df_picks

df_picks = getDistance(df_picks, df_events, stations, sac_path)

# %%
def timeSummary(df, savepath):
    p_delay = df['P_time'][df['P_time'].notna()]-df['start'][df['P_time'].notna()]
    s_delay = df['S_time'][df['S_time'].notna()]-df['start'][df['S_time'].notna()]

    fig, ax = plt.subplots()
    n, bins, patches = plt.hist(p_delay, bins=30, alpha=0.7, rwidth=0.85)
    mu = np.mean(p_delay)
    sigma = np.std(p_delay)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Arrival delay from event start (s)')
    plt.ylabel('Frequency')
    plt.title('P Arrival Delay')
    textstr = '\n'.join((
        r'$n=%.0f$' % (len(p_delay),),
        r'$\mu=%.4f$' % (mu,),
        r'$\mathrm{median}=%.4f$' % (round(np.median(p_delay), 4),),
        r'$\sigma=%.4f$' % (sigma,)))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='square,pad=.6', facecolor='lightgrey', edgecolor='black', alpha=1))
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(savepath)
    plt.show()

    fig, ax = plt.subplots()
    n, bins, patches = plt.hist(s_delay, bins=30, alpha=0.7, rwidth=0.85, facecolor='#ff7f0e')
    mu = np.mean(s_delay)
    sigma = np.std(s_delay)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Arrival delay from event start (s)')
    plt.ylabel('Frequency')
    plt.title('S Arrival Delay')
    textstr = '\n'.join((
        r'$n=%.0f$' % (len(s_delay),),
        r'$\mu=%.4f$' % (mu,),
        r'$\mathrm{median}=%.4f$' % (round(np.median(s_delay), 4),),
        r'$\sigma=%.4f$' % (sigma,)))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='square,pad=.6', facecolor='lightgrey', edgecolor='black', alpha=1))
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(savepath)
    plt.show()


timeSummary(df, plot_path)
# %%


def summary(df_picks, p_out, s_out, savepath):
    pssr = np.nansum([i ** 2 for i in df_picks['P_res'][df_picks['P_inrange']]])
    sssr = np.nansum([i ** 2 for i in df_picks['S_res'][df_picks['S_inrange']]])

    Np = df_picks['P_res'][df_picks['P_inrange']].count()
    Ns = df_picks['S_res'][df_picks['S_inrange']].count()

    prms = np.sqrt(pssr / (Np - 1))
    srms = np.sqrt(sssr / (Ns - 1))

    rms = np.sqrt((pssr + sssr) / (Np + Ns - 1))

    with open(os.path.join(savepath, "summary.txt"), "a") as f:
        print("=================================================================================", file=f)
        print(datetime.datetime.now(), file=f)

        print("Number of arrivals = ", len(df_picks), file=f)
        print("RMS = ", rms, file=f)
        print("P-RMS = ", prms, file=f)
        print("S-RMS = ", srms, file=f)
        print("---------------------------------------------------------------------------------", file=f)
        print("P picks = {}   ({}%)".format(Np, (Np / len(df_picks)) * 100), file=f)
        print("S picks = {}   ({}%)".format(Ns, (Ns / len(df_picks)) * 100), file=f)
        print("Mean P residual = ", np.nanmean(df_picks['P_res'][df_picks['P_inrange']]), file=f)
        print("Mean S residual = ", np.nanmean(df_picks['S_res'][df_picks['S_inrange']]), file=f)
        print("Median P residual = ", np.nanmedian(df_picks['P_res'][df_picks['P_inrange']]), file=f)
        print("Median S residual = ", np.nanmedian(df_picks['S_res'][df_picks['S_inrange']]), file=f)
        print("P outliers excluded = {}".format(len(p_out)), file=f)
        print("S outliers excluded = {}".format(len(s_out)), file=f)
        print("---------------------------------------------------------------------------------", file=f)
        print("PARAMETERS", file=f)
        print("P threshold = ", df_picks['P_thresh'].iloc[0], file=f)
        print("S threshold = ", df_picks['S_thresh'].iloc[0], file=f)
        print("tmin = ", df_picks['tmin'].iloc[0], file=f)
        print("tmax = ", df_picks['tmax'].iloc[0], file=f)
        print("pick method = ", df_picks['pick_method'].iloc[0], file=f)
        print("residual outlier method = ", df_picks['ol_method'].iloc[0], file=f)
        print("vps outlier method = {}".format(df_picks['vps_ol_method'].iloc[0]), file=f)
        print("filter method = {}".format(df_picks['filter_method'].iloc[0]), file=f)
        print("=================================================================================", file=f)

        print(datetime.datetime.now())

        print("Number of arrivals = ", len(df_picks))
        print("RMS = ", rms)
        print("P-RMS = ", prms)
        print("S-RMS = ", srms)
        print("---------------------------------------------------------------------------------")
        print("P picks = {}   ({}%)".format(Np, (Np / len(df_picks)) * 100))
        print("S picks = {}   ({}%)".format(Ns, (Ns / len(df_picks)) * 100))
        print("Mean P residual = ", np.nanmean(df_picks['P_res'][df_picks['P_inrange']]))
        print("Mean S residual = ", np.nanmean(df_picks['S_res'][df_picks['S_inrange']]))
        print("Median P residual = ", np.nanmedian(df_picks['P_res'][df_picks['P_inrange']]))
        print("Median S residual = ", np.nanmedian(df_picks['S_res'][df_picks['S_inrange']]))
        print("P outliers excluded = {}".format(len(p_out)))
        print("S outliers excluded = {}".format(len(s_out)))
        print("---------------------------------------------------------------------------------")
        print("PARAMETERS")
        print("P threshold = ", df_picks['P_thresh'].iloc[0])
        print("S threshold = ", df_picks['S_thresh'].iloc[0])
        print("tmin = ", df_picks['tmin'].iloc[0])
        print("tmax = ", df_picks['tmax'].iloc[0])
        print("pick method = ", df_picks['pick_method'].iloc[0])
        print("residual outlier method = ", df_picks['ol_method'].iloc[0])
        print("vps outlier method = {}".format(df_picks['vps_ol_method'].iloc[0]))
        print("filter method = {}".format(df_picks['filter_method'].iloc[0]))
        print("=================================================================================")

    print("Summary information saved to ", os.path.join(savepath, 'summary.txt'))


summary(df_picks, p_out, s_out, plot_path)


def Histogram(df_picks, phase, p_out, s_out, vps_out, plot_path, Methods, method):
    fig, ax = plt.subplots()
    if phase == 'P':
        c = '#1f77b4'
        d = df_picks['P_res'][df_picks['P_inrange']]
        ol = len(p_out)
        rem = len(df_picks) - len(d)
        path = os.path.join(plot_path, 'p_hist.png')
        plt.xlim(-method[1], method[1])
    elif phase == 'S':
        c = '#ff7f0e'
        d = df_picks['S_res'][df_picks['S_inrange']]
        ol = len(s_out)
        rem = len(df_picks) - len(d)
        path = os.path.join(plot_path, 's_hist.png')
        plt.xlim(-method[1], method[1])
    elif phase == 'vps':
        c = '#d62728'
        ol = '{} IQR'.format(len(vps_out))
        rem = len(vps_out)
        d = df_picks['vps'][df_picks['vps_inrange']]
        path = os.path.join(plot_path, 'vps_hist.png')
    else:
        print("Invalid phase entry: phase= ('P', 'S', 'vps')")

    n, bins, patches = plt.hist(x=d, bins='auto', color=c,
                                alpha=0.7, rwidth=0.85)
    mu = np.nanmean(d)
    sigma = np.nanstd(d)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Residual time (s)')
    plt.ylabel('Frequency*')
    plt.title('Histogram of {}-pick Residuals (expected-observed)'.format(phase))
    textstr = '\n'.join((
        r'$n=%.0f$' % (np.count_nonzero(~np.isnan(d)),),
        r'$\mu=%.4f$' % (mu,),
        r'$\mathrm{median}=%.4f$' % (round(np.nanmedian(d), 4),),
        r'$\sigma=%.4f$' % (sigma,)))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='square,pad=.6', facecolor='lightgrey', edgecolor='black', alpha=1))
    plt.annotate('*method={}: {} '.format(method, Methods[method[0]]), (0, 0), (0, -40),
                 xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
    plt.annotate('{} total arrivals removed, {} outliers removed'.format(rem, ol), (0, 0), (0, -50),
                 xycoords='axes fraction', textcoords='offset points', va='top', style='italic', fontsize=9)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(path)
    plt.show()


Histogram(df_picks, 'P', p_out, s_out, vps_out, plot_path, methods, outlier_method)
Histogram(df_picks, 'S', p_out, s_out, vps_out, plot_path, methods, outlier_method)
Histogram(df_picks, 'vps', p_out, s_out, vps_out, plot_path, methods, outlier_method)


# %%
def scatterPlot(df_picks, plot_path, option, method):
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

    if option == 'boxplot':
        gs1 = fig.add_gridspec(ncols=1, nrows=1, top=0.93, bottom=0.3, left=0.1, right=0.95)
        gs2 = fig.add_gridspec(ncols=1, nrows=2, top=0.22, bottom=0.02, left=0.1, right=0.95, hspace=0.02)

        ax1 = plt.subplot(gs1[0], xlim=(0, method[1]), ylim=(0, 1))
        ax1.grid(axis='both', alpha=0.4)
        ax1.set(ylabel='PhaseNet probability*', xlabel='abs(Residual time) (s)')

        pl, = ax1.plot(px, p_intercept + p_slope * px, color='#333333', linestyle='dashdot', lw=2.4,
                       label="y=%.2fx+%.2f" % (p_slope, p_intercept))
        sl, = ax1.plot(sx, s_intercept + s_slope * sx, color='#333333', linestyle='dotted', lw=2.65,
                       label="y=%.2fx+%.2f" % (s_slope, s_intercept))
        pscat = ax1.scatter(p_res_np, p_prob_np)
        pscat.set_label('P')
        sscat = ax1.scatter(s_res_np, s_prob_np)
        sscat.set_label('S')
        leg = ax1.legend([pscat, pl, sscat, sl],
                         ['P', "y=%.2fx+%.2f" % (p_slope, p_intercept), 'S', "y=%.2fx+%.2f" % (s_slope, s_intercept)],
                         loc='upper right')
        leg.get_frame().set_edgecolor('#262626')

        ax2 = plt.subplot(gs2[0], frame_on=False, xlim=(0, method[1]))
        ax2.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.boxplot(p_res_np, vert=False, whis=(0, 100), patch_artist=True,
                    boxprops=dict(facecolor='#1f77b4', color='k'),
                    medianprops=dict(color='k'),
                    )

        ax3 = plt.subplot(gs2[1], frame_on=False, xlim=(0, method[1]))
        ax3.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax3.boxplot(s_res_np, vert=False, whis=(0, 100), patch_artist=True,
                    boxprops=dict(facecolor='#ff7f0e', color='k'),
                    medianprops=dict(color='k'),
                    )
    elif option == 'PDF':
        n, p_bins, patches = plt.hist(p_res_np, bins='auto')
        py = stats.norm.pdf(p_bins, np.mean(p_res_np), np.std(p_res_np))

        n, s_bins, patches = plt.hist(s_res_np, bins='auto')
        sy = stats.norm.pdf(s_bins, np.mean(s_res_np), np.std(s_res_np))

        gs1 = fig.add_gridspec(ncols=1, nrows=1, top=0.93, bottom=0.23, left=0.1, right=0.95)
        gs2 = fig.add_gridspec(ncols=1, nrows=1, top=0.13, bottom=0.035, left=0.1, right=0.95)

        ax1 = plt.subplot(gs1[0], xlim=(0, method[1]), ylim=(0, 1))
        ax1.grid(axis='both', alpha=0.4)
        ax1.set(ylabel='PhaseNet probability*', xlabel='abs(Residual time) (s)')

        pl, = ax1.plot(px, p_intercept + p_slope * px, color='#333333', linestyle='dashdot', lw=2.4,
                       label="y=%.2fx+%.2f" % (p_slope, p_intercept))
        sl, = ax1.plot(sx, s_intercept + s_slope * sx, color='#333333', linestyle='dotted', lw=2.65,
                       label="y=%.2fx+%.2f" % (s_slope, s_intercept))
        pscat = ax1.scatter(p_res_np, p_prob_np)
        pscat.set_label('P')
        sscat = ax1.scatter(s_res_np, s_prob_np)
        sscat.set_label('S')
        leg = ax1.legend([pscat, pl, sscat, sl],
                         ['P', "y=%.2fx+%.2f" % (p_slope, p_intercept), 'S', "y=%.2fx+%.2f" % (s_slope, s_intercept)],
                         loc='upper right')
        leg.get_frame().set_edgecolor('#262626')

        ax2 = plt.subplot(gs2[0], xlim=(0, method[1]))

        ax2.plot(p_bins, py, '--', linewidth=2)
        ax2.plot(s_bins, sy, '--', linewidth=2)
        plt.ylabel('PDF')

        # ax2.plot(p_hist_dist.pdf(px), linewidth=2.5, label='P PDF')
        # ax2.plot(s_hist_dist.pdf(sx), linewidth=2.5, label='S PDF')

        # ax2.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)
        # plt.setp(ax2.get_yticklabels(), visible=False)
    else:
        print("Invalid option arg: ('boxplot', 'PDF')")

    plt.savefig(os.path.join(plot_path, 'scatter.png'))

    # print("P:   y=%.6fx+(%.6f)" % (p_slope, p_intercept))
    # print("S:   y=%.6fx+(%.6f)" % (s_slope, s_intercept))
    plt.show()
    plt.close()


scatterPlot(df_picks, plot_path, 'PDF', outlier_method)


# %%


# %%
def vpsPlot(df_picks, samples, delta, option, plot_path):
    itp = df_picks['itp'][df_picks['vps_inrange']] * delta
    its = df_picks['its'][df_picks['vps_inrange']] * delta
    vps = df_picks['vps'][df_picks['vps_inrange']]

    plt.close()

    fig = plt.figure()
    fig.suptitle('PhaseNet VP/VS ratios')

    x = np.linspace(0, (samples + 100) * delta, len(vps))

    if option == 'boxplot':
        gs1 = fig.add_gridspec(ncols=1, nrows=1, top=0.93, bottom=0.2, left=0.13, right=0.95)
        gs2 = fig.add_gridspec(ncols=1, nrows=1, top=0.15, bottom=0.02, left=0.1, right=0.95)

        ax1 = plt.subplot(gs1[0], xlim=(0, (samples + 100) * delta), ylim=(0, (samples + 100) * delta))
        ax1.grid(axis='both', alpha=0.4)
        ax1.set(ylabel='S arrival (s)', xlabel='P arrival (s)')

        mean_l, = ax1.plot(x, np.mean(vps) * x, color='#333333', linestyle='dashdot', lw=2.4)
        # med_l, = ax1.plot(x, np.median(vps) * x, color='#333333', linestyle=':', lw=2.4)

        slope, intercept, r_value, p_value, std_err = stats.linregress(itp.astype('float32'), its.astype('float32'))
        fit_l, = ax1.plot(x, intercept + slope * x, color='#333333', linestyle=':', lw=2.4)

        ax1.scatter(itp, its, c='#d62728')
        leg = ax1.legend([mean_l, fit_l],
                         ["mean: y=%.2fx+%.0f" % (np.mean(vps), 0), "reg fit: y=%.2fx+%.0f" % (slope, intercept)],
                         loc='upper left')
        leg.get_frame().set_edgecolor('#262626')

        ax2 = plt.subplot(gs2[0], frame_on=False, xlim=(0, (samples + 100) * delta))
        ax2.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)

        ax2.boxplot(itp, vert=False, whis=(0, 100), patch_artist=True,
                    boxprops=dict(facecolor='#d62728', color='k'),
                    medianprops=dict(color='k'),
                    )
    elif option == 'PDF':
        n, p_bins, patches = plt.hist(itp, bins='auto', range=(0, (samples + 100) * delta))
        py = stats.norm.pdf(p_bins, np.mean(itp), np.std(itp))

        n, s_bins, patches = plt.hist(its, bins='auto', range=(0, (samples + 100) * delta))
        sy = stats.norm.pdf(s_bins, np.mean(its), np.std(its))

        gs1 = fig.add_gridspec(ncols=1, nrows=1, top=0.93, bottom=0.23, left=0.1, right=0.95)
        gs2 = fig.add_gridspec(ncols=1, nrows=1, top=0.13, bottom=0.035, left=0.1, right=0.95)

        ax1 = plt.subplot(gs1[0], xlim=(0, (samples + 100) * delta), ylim=(0, (samples + 100) * delta))
        ax1.grid(axis='both', alpha=0.4)
        ax1.set(ylabel='S arrival (s)', xlabel='P arrival (s)')

        mean_l, = ax1.plot(x, np.mean(vps) * x, color='#333333', linestyle='dashdot', lw=2.4)
        # med_l, = ax1.plot(x, np.median(vps) * x, color='#333333', linestyle=':', lw=2.4)

        slope, intercept, r_value, p_value, std_err = stats.linregress(itp.astype('float32'), its.astype('float32'))
        fit_l, = ax1.plot(x, intercept + slope * x, color='#333333', linestyle=':', lw=2.4)

        ax1.scatter(itp, its, c='#d62728')
        leg = ax1.legend([mean_l, fit_l],
                         ["mean: y=%.2fx+%.0f" % (np.mean(vps), 0), "reg fit: y=%.2fx+%.0f" % (slope, intercept)],
                         loc='upper left')
        leg.get_frame().set_edgecolor('#262626')

        ax2 = plt.subplot(gs2[0], xlim=(0, (samples + 100) * delta))
        # ax2.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)
        # plt.setp(ax2.get_yticklabels(), visible=False)

        ax2.plot(p_bins, py, '--', linewidth=2)
        ax2.plot(s_bins, sy, '--', linewidth=2)
        plt.ylabel("PDF")
    else:
        print("Invalid option arg: ('boxplot', 'PDF')")

    plt.savefig(os.path.join(plot_path, 'vps.png'))
    plt.show()
    plt.close()


vpsPlot(df_picks, n_samp, dt, 'PDF', plot_path)


# %%

# %%
def wadati(df_picks, samples, delta, option, savepath):
    itp = df_picks['itp'][df_picks['vps_inrange']] * delta
    its = df_picks['its'][df_picks['vps_inrange']] * delta
    vps = (its - itp) / itp

    plt.close()

    fig = plt.figure()
    fig.suptitle('Wadati Plot')

    gs1 = fig.add_gridspec(ncols=1, nrows=1, top=0.93, bottom=0.2, left=0.13, right=0.95)
    gs2 = fig.add_gridspec(ncols=1, nrows=1, top=0.15, bottom=0.02, left=0.13, right=0.95)

    ax1 = plt.subplot(gs1[0], xlim=(0, (samples + 100) * delta))
    ax1.grid(axis='both', alpha=0.4)
    ax1.set(ylabel='S - P time (s)', xlabel='P arrival time (s)')

    x = np.linspace(0, (samples + 100) * delta, len(vps))
    mean_l, = ax1.plot(x, np.mean(vps) * x, color='#333333', linestyle='dashdot', lw=2.4)
    # med_l, = ax1.plot(x, np.median(vps) * x, color='#333333', linestyle=':', lw=2.4)

    slope, intercept, r_value, p_value, std_err = stats.linregress(itp, its - itp)
    fit_l, = ax1.plot(x, intercept + slope * x, color='#333333', linestyle=':', lw=2.4)
    #
    ax1.scatter(itp, its - itp, c='#d62728')
    leg = ax1.legend([mean_l, fit_l],
                     ["mean: y=%.2fx+%.0f" % (np.mean(vps), 0), "reg fit: y=%.2fx+%.0f" % (slope, intercept)],
                     loc='upper left')
    leg.get_frame().set_edgecolor('#262626')

    if option == 'boxplot':
        ax2 = plt.subplot(gs2[0], frame_on=False)
        ax2.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.boxplot(itp, vert=False, whis=(0, 100), patch_artist=True,
                    boxprops=dict(facecolor='#d62728', color='k'),
                    medianprops=dict(color='k'),
                    )
    elif option == 'PDF':
        ax2 = plt.subplot(gs2[0], xlim=(0, (samples + 100) * delta), frame_on=True)
        p_hist = np.histogram(itp, bins='auto')
        p_hist_dist = stats.rv_histogram(p_hist)

        ax2.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.plot(x, p_hist_dist.pdf(x), c='#d62728', linewidth=2.5, label='P PDF')
    else:
        print("Invalid option arg: ('boxplot', 'PDF')")

    plt.savefig(os.path.join(plot_path, 'wadati.png'))
    plt.show()
    plt.close()


wadati(df_picks, n_samp, dt, 'PDF', plot_path)

# %%

def distancePlot(df_picks, option, savepath):
    fig = plt.figure()
    fig.suptitle('Distance scatter')

    xlin = np.linspace(np.min(df_picks['distance']), np.max(df_picks['distance']), len(df_picks['distance']))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_picks['distance'], abs(df_picks['P_res']))
    if option == 'boxplot':
        gs1 = fig.add_gridspec(ncols=1, nrows=1, top=0.93, bottom=0.2, left=0.13, right=0.95)
        gs2 = fig.add_gridspec(ncols=1, nrows=1, top=0.15, bottom=0.02, left=0.1, right=0.95)

        ax1 = plt.subplot(gs1[0], ylim=(0, 4))
        ax1.grid(axis='both', alpha=0.4)
        ax1.set(ylabel='abs(P_res) (s)', xlabel='distance (km)')

        ax1.scatter(df_picks['distance'], abs(df_picks['P_res']), c='#008080')
        plt.xlim(-4, 4)
        fit_l, = ax1.plot(xlin, intercept + slope * xlin, color='#333333', linestyle=':', lw=2.4)

        leg = ax1.legend([fit_l],
                         ["linear fit: y=%.2fx+%.0f" % (slope, intercept)],
                         loc='upper left')
        leg.get_frame().set_edgecolor('#262626')

        ax2 = plt.subplot(gs2[0], frame_on=False)
        ax2.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.boxplot(abs(df_picks['P_res']), vert=False, whis=(0, 100), patch_artist=True,
                    boxprops=dict(facecolor='#008080', color='k'),
                    medianprops=dict(color='k'),
                    )
    elif option == 'PDF':
        gs1 = fig.add_gridspec(ncols=1, nrows=1, top=0.93, bottom=0.23, left=0.1, right=0.95)
        gs2 = fig.add_gridspec(ncols=1, nrows=1, top=0.13, bottom=0.035, left=0.1, right=0.95)

        ax1 = plt.subplot(gs1[0])
        ax1.grid(axis='both', alpha=0.4)
        ax1.set(ylabel='abs(P_res) (s)', xlabel='distance (km)')

        ax1.scatter(df_picks['distance'], abs(df_picks['P_res']), c='#008080')
        plt.ylim(0, 4)
        fit_l, = ax1.plot(xlin, intercept + slope * xlin, color='#333333', linestyle=':', lw=2.4)

        leg = ax1.legend([fit_l],
                         ["linear fit: y=%.2fx+%.0f" % (slope, intercept)],
                         loc='upper left')
        leg.get_frame().set_edgecolor('#262626')

        ax2 = plt.subplot(gs2[0], frame_on=True)
        n, bins, patches = plt.hist(df_picks['distance'], bins='auto')
        y = stats.norm.pdf(bins, np.mean(df_picks['distance']), np.std(df_picks['distance']))

        ax2 = plt.subplot(gs2[0])
        # ax2.tick_params(axis=u'both', which=u'both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)
        # plt.setp(ax2.get_yticklabels(), visible=False)

        ax2.plot(bins, y, '--', c='#008080', linewidth=2)
        plt.ylabel("PDF")

    else:
        print("Invalid option arg: ('boxplot', 'PDF')")

    plt.savefig(os.path.join(plot_path, 'distance_scatter.png'))
    plt.show()
    plt.close()

    x = df_picks['P_res'][df_picks['P_res'].isna() == False]
    y = df_picks['distance'][df_picks['P_res'].isna() == False]
    # plt.hist2d(x, y, bins=(60, 60), range=[[-1,1], [0, 1]], weights= len(x)/ np.ones(len(x)), cmap=plt.cm.jet)

    print(max(y))
    H, xedges, yedges = np.histogram2d(x, y, range=[[-1, 1], [0, 600]], bins=(60, 60))
    row_sums = H.sum(axis=0)
    norm_rows = H / row_sums[:, np.newaxis]
    # H_norm_rows = H / H.max(axis=1, keepdims=True)
    # H_norm_rows = (H.T / np.sum(H, axis=1)).T
    # plt.hist2d()
    plt.imshow(np.fliplr(np.flipud(norm_rows.T)))

    plt.colorbar()
    plt.title("2D Histogram of Residuals by Distance")
    plt.xlabel("residual (s)")
    plt.xticks(ticks=np.linspace(0, 60, num=9,endpoint=True), labels=np.linspace(-1, 1, num=9,endpoint=True))
    plt.yticks(ticks=np.linspace(60, 0, num=9, endpoint=True), labels=np.linspace(0, 600, num=9, endpoint=True))
    plt.ylabel("distance (km)")
    #plt.xlim(-1, 1)
    plt.show()


distancePlot(df_picks, 'PDF', plot_path)
# %%