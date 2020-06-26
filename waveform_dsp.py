import pandas as pd
import numpy as np
import obspy
import matplotlib.pyplot as plt
import os

PN_pick_method = ['min_res']
outlier_method = ['over', 2]
vps_method = ['outlier']

tbegin = -30  # starttime is 30 seconds prior to origin of earthquake
tend = 100  # end time is 240 seconds after origin of earthquake
dt = 0.01
n_samp = int((tend - tbegin) / dt + 1)

channels = ['E', 'N', 'Z']

sac_source = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/'
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
    df = pd.read_pickle(os.path.join(dataset_path, 'data_log_merged.pickle'))
    df_picks = pd.read_pickle(os.path.join(outlier_path, 'filter_picks.pickle'))
    p_outliers = pd.read_pickle(os.path.join(outlier_path, 'p_outliers.pickle'))
    s_outliers = pd.read_pickle(os.path.join(outlier_path, 's_outliers.pickle'))
    vps_outliers = pd.read_pickle(os.path.join(outlier_path, 'vps_outliers.pickle'))
    return log, picks, arrivals, df, df_picks, p_outliers, s_outliers, vps_outliers


log, picks, arrivals, df, df_picks, p_outliers, s_outliers, vps_outliers = \
    initFrames(dataset_path, output_path, arrival_path)

fnames = df_picks['fname']
# fnames = 'NZ_WACZ_EH_2020p246942_13001'


def superPlotter5000(fnames, df, df_picks, t_window, focus, sac_path, output_path):
    if type(fnames) == str:
        if not fnames.endswith('.npz'):
            fnames = [fnames+'.npz']
        else:
            fnames = [fnames]

    for f in fnames:
        row = df.loc[df['fname'] == f]
        i = row.index[0]
        sac_files = row['network'].astype("string") + '_' + \
                    row['station'].astype("string") + '_' + \
                    row['channel'].astype("string") + '.SAC'

        print("fname: ", f)
        print("focus: ", focus)
        print("t window = ", t_window)
        print("P res = ", df_picks['P_res'][i])
        print("S res = ", df_picks['S_res'][i])
        print("VPS = ", df_picks['vps'][i])
        print("Pick method: ", df_picks['pick_method'][i])
        print("===================================================")

        st = obspy.read(os.path.join(sac_path, row['event_id'].iloc[0], sac_files.iloc[0]))

        PN_results = np.load(os.path.join(output_path + '/results', f))
        p_prob_data = np.column_stack(np.concatenate(PN_results['pred']))[1]
        s_prob_data = np.column_stack(np.concatenate(PN_results['pred']))[2]

        delta = st[0].stats.delta
        pm = (t_window / delta) / 2

        try:
            P_itp = (df_picks['P_time'][i] - df['start'][i]) / delta
        except TypeError:
            P_itp = 0
        try:
            S_itp = (df_picks['S_time'][i] - df['start'][i]) / delta
        except TypeError:
            S_itp = 0

        if focus == 'P_phasenet':
            if np.isnan(df_picks['itp'][i]):
                print("WARNING: PhaseNet P pick does not exist. Plotting entire wave...")
                start = 0
                end = df['samples'][i] - 1
            else:
                start = int(df_picks['itp'][i] - pm)
                end = int(df_picks['itp'][i] + pm)
        elif focus == 'P_time':
            if P_itp == 0:
                print("WARNING: GeoNet P arrival does not exist. Plotting entire wave...")
                start = 0
                end = df['samples'][i] - 1
            else:
                start = int(P_itp - pm)
                end = int(P_itp + pm)
        elif focus == 'P_both':
            if P_itp == 0:
                print("WARNING: GeoNet P arrival does not exist. Plotting entire wave...")
                start = 0
                end = df['samples'][i] - 1
            elif np.isnan(df_picks['itp'][i]):
                print("WARNING: PhaseNet P pick does not exist. Plotting entire wave...")
                start = 0
                end = df['samples'][i] - 1
            else:
                center = P_itp - (P_itp - df_picks['itp'][i]) / 2
                start = int(center - pm)
                end = int(center + pm)

                if P_itp >= center + pm or P_itp <= center - pm:
                    print("WARNING: focus is set to both but picks fall outside time window. Plotting entire wave...")
                    start = 0
                    end = df['samples'][i] - 1
        elif focus == 'S_phasenet':
            if np.isnan(df_picks['its'][i]):
                print("WARNING: PhaseNet S pick does not exist. Plotting entire wave...")
                start = 0
                end = df['samples'][i] - 1
            else:
                start = int(df_picks['its'][i] - pm)
                end = int(df_picks['its'][i] + pm)
        elif focus == 'S_time':
            if S_itp == 0:
                print("WARNING: GeoNet S arrival does not exist. Plotting entire wave...")
                start = 0
                end = df['samples'][i] - 1
            else:
                start = int(S_itp - pm)
                end = int(S_itp + pm)
        elif focus == 'S_both':
            if S_itp == 0:
                print("WARNING: GeoNet S arrival does not exist. Plotting entire wave...")
                start = 0
                end = df['samples'][i] - 1
            elif np.isnan(df_picks['its'][i]):
                print("WARNING: PhaseNet S pick does not exist. Plotting entire wave...")
                start = 0
                end = df['samples'][i] - 1
            else:
                center = S_itp - (S_itp - df_picks['its'][i]) / 2
                start = int(center - pm)
                end = int(center + pm)

                if S_itp >= center + pm or S_itp <= center - pm:
                    print("WARNING: focus is set to both but picks fall outside time window. Plotting entire wave...")
                    start = 0
                    end = df['samples'][i] - 1
        elif focus == 'all':
            start = 0
            end = df['samples'][i] - 1
        else:
            start = int(focus / delta - pm)
            end = int(focus / delta + pm)

        if start < 0:
            print("WARNING: Time window begins before trace data. Showing plots with start = 0")
            start = 0
        if end > df['samples'][i]:
            print("WARNING: Time window ends after trace data. Showing plots with end = max-samples ({})"
                  .format(df_picks['itp'][i]))
            end = df['samples'][i]

        times = st[0].times()  # seconds
        data = [st[0].data, st[1].data, st[2].data]  # E, N, Z

        box = dict(boxstyle='round', facecolor='white', alpha=1)
        text_loc = [0.05, 0.77]

        plt.figure()
        ax1 = plt.subplot(411)
        ax1.set_xlim(times[start], times[end])
        plt.plot(times[start:end], data[0][start:end], 'k', label='E', linewidth=0.5)  # for seconds

        tmp_min = np.min(data[0][start:end])
        tmp_max = np.max(data[0][start:end])

        for j in range(len(df['itp'][i])):
            if j == 0:
                plt.plot([df['itp'][i][j] * delta, df['itp'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#1f77b4', linewidth=1.5)
            else:
                plt.plot([df['itp'][i][j] * delta, df['itp'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#1f77b4', linewidth=1.5)
        for j in range(len(df['its'][i])):
            if j == 0:
                plt.plot([df['its'][i][j] * delta, df['its'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#ff7f0e', linewidth=1.5)
            else:
                plt.plot([df['its'][i][j] * delta, df['its'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#ff7f0e', linewidth=1.5)

        if not np.isnan(df_picks['itp'][i]):
            plt.plot([df_picks['itp'][i] * delta, df_picks['itp'][i] * delta], [tmp_min, tmp_max],
                     c='#1f77b4', linewidth=1.5)
        if not np.isnan(df_picks['its'][i]):
            plt.plot([df_picks['its'][i] * delta, df_picks['its'][i] * delta], [tmp_min, tmp_max],
                     c='#ff7f0e', linewidth=1.5)

        plt.plot([P_itp * delta, P_itp * delta], [tmp_min, tmp_max], c='slategray', linewidth=1.5)
        plt.plot([S_itp * delta, S_itp * delta], [tmp_min, tmp_max], c='r', linewidth=1.5)

        plt.ylabel('Amplitude')
        plt.legend(loc='upper right', fontsize='small')
        plt.gca().set_xticklabels([])
        plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
                 transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

        ax2 = plt.subplot(412)
        ax2.set_xlim(times[start], times[end])
        plt.plot(st[1].times()[start:end], data[1][start:end], 'k', label='N', linewidth=0.5)

        tmp_min = np.min(data[1][start:end])
        tmp_max = np.max(data[1][start:end])

        for j in range(len(df['itp'][i])):
            if j == 0:
                plt.plot([df['itp'][i][j] * delta, df['itp'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#1f77b4', linewidth=1.5)
            else:
                plt.plot([df['itp'][i][j] * delta, df['itp'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#1f77b4', linewidth=1.5)
        for j in range(len(df['its'][i])):
            if j == 0:
                plt.plot([df['its'][i][j] * delta, df['its'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#ff7f0e', linewidth=1.5)
            else:
                plt.plot([df['its'][i][j] * delta, df['its'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#ff7f0e', linewidth=1.5)

        if not np.isnan(df_picks['itp'][i]):
            plt.plot([df_picks['itp'][i] * delta, df_picks['itp'][i] * delta], [tmp_min, tmp_max],
                     c='#1f77b4', linewidth=1.5)
        if not np.isnan(df_picks['its'][i]):
            plt.plot([df_picks['its'][i] * delta, df_picks['its'][i] * delta], [tmp_min, tmp_max],
                     c='#ff7f0e', linewidth=1.5)

        plt.plot([P_itp * delta, P_itp * delta], [tmp_min, tmp_max], c='slategray', linewidth=1.5)
        plt.plot([S_itp * delta, S_itp * delta], [tmp_min, tmp_max], c='r', linewidth=1.5)

        plt.ylabel('Amplitude')
        plt.legend(loc='upper right', fontsize='small')
        plt.gca().set_xticklabels([])
        plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
                 transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

        ax3 = plt.subplot(413)
        ax3.set_xlim(times[start], times[end])
        plt.plot(st[2].times()[start:end], data[2][start:end], 'k', label='Z', linewidth=0.5)

        tmp_min = np.min(data[2][start:end])
        tmp_max = np.max(data[2][start:end])

        for j in range(len(df['itp'][i])):
            if j == 0:
                plt.plot([df['itp'][i][j] * delta, df['itp'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#1f77b4', linewidth=1.5)
            else:
                plt.plot([df['itp'][i][j] * delta, df['itp'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#1f77b4', linewidth=1.5)
        for j in range(len(df['its'][i])):
            if j == 0:
                plt.plot([df['its'][i][j] * delta, df['its'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#ff7f0e', linewidth=1.5)
            else:
                plt.plot([df['its'][i][j] * delta, df['its'][i][j] * delta], [tmp_min, tmp_max],
                         '--', c='#ff7f0e', linewidth=1.5)

        if not np.isnan(df_picks['itp'][i]):
            plt.plot([df_picks['itp'][i] * delta, df_picks['itp'][i] * delta], [tmp_min, tmp_max],
                     c='#1f77b4', linewidth=1.5)
        if not np.isnan(df_picks['its'][i]):
            plt.plot([df_picks['its'][i] * delta, df_picks['its'][i] * delta], [tmp_min, tmp_max],
                     c='#ff7f0e', linewidth=1.5)

        plt.plot([P_itp * delta, P_itp * delta], [tmp_min, tmp_max], c='slategray', linewidth=1.5)
        plt.plot([S_itp * delta, S_itp * delta], [tmp_min, tmp_max], c='r', linewidth=1.5)

        plt.ylabel('Amplitude')
        plt.legend(loc='upper right', fontsize='small')
        plt.gca().set_xticklabels([])
        plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
                 transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

        ax4 = plt.subplot(414)
        ax4.set_xlim(times[start], times[end])
        plt.plot(times[start:end], p_prob_data[start:end], '--g', label='$\hat{P}$', linewidth=0.5)
        plt.plot(times[start:end], s_prob_data[start:end], '-.m', label='$\hat{S}$', linewidth=0.5)

        for j in range(len(df['itp'][i])):
            if j == 0:
                plt.plot([df['itp'][i][j] * delta, df['itp'][i][j] * delta], [0, 1],
                         '--', c='#1f77b4', linewidth=1.5)
            else:
                plt.plot([df['itp'][i][j] * delta, df['itp'][i][j] * delta], [0, 1],
                         '--', c='#1f77b4', linewidth=1.5)
        for j in range(len(df['its'][i])):
            if j == 0:
                plt.plot([df['its'][i][j] * delta, df['its'][i][j] * delta], [0, 1],
                         '--', c='#ff7f0e', linewidth=1.5)
            else:
                plt.plot([df['its'][i][j] * delta, df['its'][i][j] * delta], [0, 1],
                         '--', c='#ff7f0e', linewidth=1.5)

        if not np.isnan(df_picks['itp'][i]):
            plt.plot([df_picks['itp'][i] * delta, df_picks['itp'][i] * delta], [0, 1], '#1f77b4', linewidth=1.5)
        if not np.isnan(df_picks['its'][i]):
            plt.plot([df_picks['its'][i] * delta, df_picks['its'][i] * delta], [0, 1], c='#ff7f0e', linewidth=1.5)

        plt.plot([P_itp * delta, P_itp * delta], [0, 1], c='slategray', linewidth=1.5)
        plt.plot([S_itp * delta, S_itp * delta], [0, 1], c='r', linewidth=1.5)

        plt.plot(times[start:end], [df_picks['P_thresh'][i]] * len(times[start:end]), ':', linewidth=0.25)
        plt.plot(times[start:end], [df_picks['S_thresh'][i]] * len(times[start:end]), ':', linewidth=0.25)

        plt.ylim([-0.05, 1.05])
        plt.text(text_loc[0], text_loc[1], '(iv)', horizontalalignment='center',
                 transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
        plt.legend(loc='upper right', fontsize='small')
        plt.xlabel('Time (s)')
        plt.ylabel('Probability')

        plt.text(0.02, 0.02, "fname: {}".format(f), fontsize=10, transform=plt.gcf().transFigure)
        plt.text(0.6, 0.02, "| P-res = {}   |   S-res = {} |".
                 format(round(df_picks['P_res'][i], 2), round(df_picks['S_res'][i]), 2),
                 fontsize=10, transform=plt.gcf().transFigure)

        plt.tight_layout()
        plt.gcf().align_labels()

        plt.savefig(os.path.join('/Volumes/WMEL/Kaikoura Plots', os.path.splitext(f)[0]+'.png'))
        plt.show()


superPlotter5000(fnames, df, df_picks, 5, 'S_phasenet', sac_source, output_path)
