# Documentation:
# https://numpy.org/doc/stable/reference/generated/numpy.savez.html

import obspy
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/2017p357939/NZ_AKCZ_EHE.SAC'


def plotTrace(sac_file_path):
    tr = obspy.read(sac_file_path)
    tr.plot()


plotTrace(path)


def filterTrace(sac_file_path):
    st = obspy.read(sac_file_path)
    tr = st[0]
    tr_filt = tr.copy()

    tr_filt.filter('bandpass', freqmin=2, freqmax=10, corners=2, zerophase=True)

    t = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)
    plt.subplot(211)
    plt.plot(t, tr.data, 'k')
    plt.ylabel('Raw Data')
    plt.subplot(212)
    plt.plot(t, tr_filt.data, 'k')
    plt.ylabel('Lowpassed Data')
    plt.xlabel('Time [s]')
    plt.suptitle(tr.stats.starttime)
    plt.show()


filterTrace(path)


def streamInfo(sac_file_path):
    st = obspy.read(sac_file_path)
    sac_length = len(st)
    print("Traces in SAC file: ", sac_length)

    stat_list = []
    data_len_list = []
    data_list = []
    for tr in range(sac_length):
        trace = st[tr]
        data_len = len(trace.data)
        tr_data = trace.data
        stat = trace.stats

        print("Number of data points: ", data_len)
        print("Trace  {} statistics: \n".format(tr + 1), stat)

        stat_list.append(stat)
        data_len_list.append(data_len)
        data_list.append(tr_data)
    return stat_list, data_len_list, data_list, sac_length


stats, data_length, data, sac_length = streamInfo(path)


def getTime(stats):
    time_window = []
    for tr in range(len(stats)):
        start = stats[tr].starttime
        end = stats[tr].endtime
        time_window.append(end - start)
    return time_window


time = getTime(stats)
print("Duration of trace (sec): ", time)

def npzConverter(sac_file_path):
    st = obspy.read(sac_file_path)
    tr = st[0]

    np.savez("/Users/Lenni/Downloads/outfile", tr.data)

npzConverter(path)


npzfile = np.load("/Users/Lenni/Downloads/outfile.npz")
npzfile.files
npzfile['arr_0']

