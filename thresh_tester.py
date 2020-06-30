import datetime
import obspy
import numpy as np
import os
import pandas as pd
import shlex

test_name = 'filter4'
test_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset/FIlter tests'

PN_pick_method = ['min_res']
outlier_method = ['over', 2]
vps_method = ['outlier']

tbegin = -30  # starttime is 30 seconds prior to origin of earthquake
tend = 100  # end time is 240 seconds after origin of earthquake
dt = 0.01
n_samp = int((tend - tbegin) / dt + 1)

threshold = np.linspace(0.05, 1, 20)


dataset_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset'
outlier_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset/Outliers'
output_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/PhaseNet/output'
arrival_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/Arrival.pickle'
plot_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset/Plots'

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

def initFrames(dataset_path, output_path, arrival_path):
    print("Initializing dataframes...")
    log = pd.read_csv(os.path.join(dataset_path, 'data_log.csv'))
    picks = pd.read_csv(os.path.join(output_path, 'picks.csv'))
    arrivals = pd.read_pickle(arrival_path)
    return log, picks, arrivals


log, picks, arrivals = initFrames(dataset_path, output_path, arrival_path)


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

def timeConverter(df):
    print("Converting arrivals to UTC DateTime...")
    for col in ['start', 'end']:
        df[col] = [obspy.UTCDateTime(x) for x in df[col]]
    return df


log = timeConverter(log)


def thresholder(picks, thresh):
    for row in range(len(picks)):
        p_idx = [i for i in range(len(picks['tp_prob'][row])) if picks['tp_prob'][row][i] < thresh]
        for index in sorted(p_idx, reverse=True):
            del picks['tp_prob'][row][index]
            del picks['itp'][row][index]

        s_idx = [i for i in range(len(picks['ts_prob'][row])) if picks['ts_prob'][row][i] < thresh]
        for index in sorted(s_idx, reverse=True):
            del picks['ts_prob'][row][index]
            del picks['its'][row][index]
    return picks


def pick2time(df):
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


def resCalculator(df):
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


def picker(df, p_thresh, s_thresh, method, savepath):
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
        for row in range(len(df)):
            fname.append(df['fname'][row])
            if not df['itp'][row]:
                p_pick.append(np.nan)
                p_res.append(np.nan)
                p_prob.append(np.nan)
                p_empty_count += 1
            elif len(df['itp'][row]) == 1:
                p_pick.append(df['P_phasenet'][row][0])
                p_prob.append(df['tp_prob'][row][0])
                try:
                    p_res.append(df['P_residual'][row][0])
                except IndexError:
                    p_res.append(np.nan)
                    pass
            else:
                if max(df['tp_prob'][row]) > 0.75:
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

    vps = []
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

    df_picks = pd.merge(df_picks, df[['event_id', 'network', 'station', 'P_time', 'S_time', 'filter_method', 'fname']],
                        how='left', on='fname')
    df_picks = df_picks[
        ["event_id", "network", "station", "P_time", "P_phasenet", "P_res", "P_prob", "itp", "P_thresh",
         "S_time", "S_phasenet", "S_res", "S_prob", "its", "S_thresh",
         "vps", "pick_method", "filter_method", "fname"]].sort_values(['event_id', 'station'])


    return df_picks, p_empty_count, s_empty_count

def outliers(df_picks, method, savepath):
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
         "S_time", "S_phasenet", "S_res", "S_prob", "its", "S_inrange",  "S_thresh",
         "vps", "pick_method", "ol_method", "filter_method", "fname"]]

    df_picks.to_pickle(os.path.join(savepath, "filter_picks.pickle"))
    df_picks.to_csv(os.path.join(savepath, "filter_picks.csv"), index=False)
    return df_picks, p_outliers, s_outliers

def vpsOutliers(df_picks, p_out, s_out, method, savepath):
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
                                 "S_time", "S_phasenet", "S_res", "S_prob", "its", "S_inrange",
                                 "vps", "vps_inrange", "pick_method", "ol_method", "filter_method", "fname"]]

    vps_outliers.to_pickle(os.path.join(savepath, "vps_outliers.pickle"))
    vps_outliers.to_csv(os.path.join(savepath, "vps_outliers.csv"), index=False)

    df_picks = df_picks[
        ["event_id", "network", "station", "P_time", "P_phasenet", "P_res", "P_prob", "itp", "P_inrange", "P_thresh",
         "S_time", "S_phasenet", "S_res", "S_prob", "its", "S_inrange", "S_thresh",
         "vps", "vps_inrange", "pick_method", "ol_method", "vps_ol_method", "filter_method", "fname"]]
    return df_picks, vps_outliers

def summary(df_picks, pick_method, outlier_method, vps_method, p_thresh, s_thresh, savepath):
    pssr = np.nansum([i ** 2 for i in df_picks['P_res'][df_picks['P_inrange']]])
    sssr = np.nansum([i ** 2 for i in df_picks['S_res'][df_picks['S_inrange']]])

    Np = np.count_nonzero(~np.isnan(df_picks['P_res'][df_picks['P_inrange']]))
    Ns = np.count_nonzero(~np.isnan(df_picks['S_res'][df_picks['S_inrange']]))

    prms = np.sqrt(pssr / (Np - 1))
    srms = np.sqrt(sssr / (Ns - 1))

    rms = np.sqrt((pssr + sssr) / (Np + Ns - 1))

    with open(os.path.join(savepath, "summary.txt"), "a") as f:
        print("=================================================================================", file=f)
        print(datetime.datetime.now(), file=f)

        print("Number of arrivals = ", len(df_picks), file=f)
        print("RMS = ", rms, file=f)
        print("Mean P residual = ", np.nanmean(df_picks['P_res'][df_picks['P_inrange']]), file=f)
        print("Mean S residual = ", np.nanmean(df_picks['S_res'][df_picks['S_inrange']]), file=f)
        print("---------------------------------------------------------------------------------", file=f)
        print("P picks = {}   ({}%)".format(df_picks['P_phasenet'].count(),
                                            (df_picks['P_phasenet'].count()/len(df_picks))*100), file=f)
        print("S picks = {}   ({}%)".format(df_picks['S_phasenet'].count(),
                                            (df_picks['S_phasenet'].count()/len(df_picks))*100), file=f)
        print("P-RMS = ", prms, file=f)
        print("S-RMS = ", srms, file=f)
        print("P outliers excluded = {}".format(len(p_out)), file=f)
        print("S outliers excluded = {}".format(len(s_out)), file=f)
        print("---------------------------------------------------------------------------------", file=f)
        print("PARAMETERS", file=f)
        print("P threshold = ", p_thresh, file=f)
        print("S threshold = ", s_thresh, file=f)
        print("pick method = ", pick_method, file=f)
        print("residual outlier method = ", outlier_method, file=f)
        print("vps outlier method = {}".format(vps_method), file=f)
        print("filter method = {}".format(df_picks['filter_method'][0]), file=f)
        print("=================================================================================", file=f)

        print(datetime.datetime.now())

        print("Number of arrivals = ", len(df_picks))
        print("RMS = ", rms)
        print("Mean P residual = ", np.nanmean(df_picks['P_res'][df_picks['P_inrange']]))
        print("Mean S residual = ", np.nanmean(df_picks['S_res'][df_picks['S_inrange']]))
        print("---------------------------------------------------------------------------------")
        print("P picks = {}   ({}%)".format(df_picks['P_phasenet'].count(),
                                            (df_picks['P_phasenet'].count() / len(df_picks)) * 100))
        print("S picks = {}   ({}%)".format(df_picks['S_phasenet'].count(),
                                            (df_picks['S_phasenet'].count() / len(df_picks)) * 100))
        print("P-RMS = ", prms)
        print("S-RMS = ", srms)
        print("P outliers excluded = {}".format(len(p_out)))
        print("S outliers excluded = {}".format(len(s_out)))
        print("---------------------------------------------------------------------------------")
        print("PARAMETERS")
        print("P threshold = ", p_thresh)
        print("S threshold = ", s_thresh)
        print("pick method = ", pick_method)
        print("residual outlier method = ", outlier_method)
        print("vps outlier method = {}".format(vps_method))
        print("filter method = {}".format(df_picks['filter_method'][0]))
        print("=================================================================================")

N = []
rms_l = []
prms_l = []
srms_l = []
p_mean = []
s_mean = []
nP = []
nS = []
p_median = []
s_median = []
P_out = []
S_out = []
for t in threshold:
    picks = thresholder(picks, t)

    df = pd.merge(log, picks, how='left', on=['fname'])
    df = pd.merge(df, arrivals[["event_id", "station", "channel", "network", "P_time", "S_time"]],
                  how='left', on=['event_id', 'station', 'network', 'channel'])

    df = pick2time(df)
    df = resCalculator(df)
    headers = ['network', 'event_id', 'station', 'channel', 'samples', 'delta', 'start', 'end', 'P_residual', 'P_time',
               'P_phasenet', 'tp_prob', 'itp', 'S_residual', 'S_time', 'S_phasenet', 'ts_prob', 'its',
               'filter_method', 'fname']
    df = df[headers].sort_values(['event_id', 'station'])

    df_picks, p_empty_count, s_empty_count = picker(df, t, t, PN_pick_method[0], outlier_path)

    df_picks, p_out, s_out = outliers(df_picks, outlier_method, outlier_path)
    df_picks, vps_out = vpsOutliers(df_picks, p_out, s_out, vps_method, outlier_path)

    summary(df_picks, PN_pick_method, outlier_method, vps_method, t, t, plot_path)

    pssr = np.nansum([i ** 2 for i in df_picks['P_res'][df_picks['P_inrange']]])
    sssr = np.nansum([i ** 2 for i in df_picks['S_res'][df_picks['S_inrange']]])

    Np = np.count_nonzero(~np.isnan(df_picks['P_res'][df_picks['P_inrange']]))
    Ns = np.count_nonzero(~np.isnan(df_picks['S_res'][df_picks['S_inrange']]))

    prms = np.sqrt(pssr / (Np - 1))
    srms = np.sqrt(sssr / (Ns - 1))

    rms = np.sqrt((pssr + sssr) / (Np + Ns - 1))

    N.append(len(df_picks))
    rms_l.append(rms)

    prms_l.append(prms)
    srms_l.append(srms)

    nP.append(df_picks['P_phasenet'].count())
    nS.append(df_picks['S_phasenet'].count())

    p_mean.append(np.nanmean(df_picks['P_res'][df_picks['P_inrange']]))
    s_mean.append(np.nanmean(df_picks['S_res'][df_picks['S_inrange']]))

    p_median.append(np.nanmedian(df_picks['P_res'][df_picks['P_inrange']]))
    s_median.append(np.nanmedian(df_picks['S_res'][df_picks['S_inrange']]))

    P_out.append(len(p_out))
    S_out.append(len(s_out))

sum_dict = {
    "prob_threshold": threshold,
    "n_files": N,
    "RMS": rms_l,
    "P_rms": prms_l,
    "S_rms": srms_l,
    "nP": nP,
    "nS": nS,
    "P_mean_res": p_mean,
    "S_mean_res": s_mean,
    "P_median_res": p_median,
    "S_median_res": s_median,
    "nP_out": P_out,
    "nS_out": S_out,
    "PN_pick_method": [PN_pick_method]*len(threshold),
    "outlier_method": [outlier_method]*len(threshold),
    "filter_method": [df_picks['filter_method'][0]]*len(threshold)
}

df_sum = pd.DataFrame.from_dict(sum_dict)
df_sum.to_pickle(os.path.join(test_path, test_name+".pickle"))
df_sum.to_csv(os.path.join(test_path, test_name+".csv"), index=False)




