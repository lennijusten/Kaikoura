# npz file writer
# Author: Lennart Justen
# Last revision: 6/2/20

# Description:
# This script takes a directory path and converts the contents into a three channel .npz file.
# Only complete sets of E,N,Z channels from a single instrument with the same number of
# samples are written. Finally the .npz reader opens a single npz file to check its contents.

# Documentation
# https://numpy.org/doc/stable/reference/generated/numpy.savez.html
# https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.select.html#obspy.core.stream.Stream.select

import obspy
import numpy as np
import os
import csv

# FOR PREDICTIONS
# todo 1) gather all the traces (3 channels) from the same station
# todo 2) Find and save the data length and check that all traces are same length
# todo 3) compile the three channels into a single .npz file with data length apparent
# todo 4) make sure traces and npz files are linkable to meta data and a data length


# FOR TRAINING
# todo 1) add expert picks for p and s waves into seperate col. in the npz file and channels!

# ACTUAL
# todo check additional requirements for phasenet (CSV and samples)
# todo write logfile to event folder
# todo automate for multiple folders (add extra wildcard in path to access all event folders)


dir_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Traces/2017p357939/*.SAC'

st = obspy.read(dir_path)

directions = ['N', 'E', 'Z']
chan_list = ['EH?']  # add channels in three letter format. Script will look for all E,N,Z channels

npz_save_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset/NPZ'
dataset_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset'

station = []
for tr in st:
    station.append(tr.stats.station)

unique_station = list(set(station))
print("Number of unique stations: ", len(unique_station))

total = 0
incomplete_count = 0
len_count = 0
len_sta = []
overfull_count = 0
overfull_sta = []


def stationChannels(stream):
    channel = []
    short_cha = []
    for tr in stream:
        channel.append(tr.stats.channel)
        short_cha.append(tr.stats.channel[:-1])

    uniq_channels = list(set(short_cha))
    print("Number of instruments: ", len(uniq_channels))

    print("Channels in stream: ")
    print(channel)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    pass


for sta in unique_station:
    set_count = 0
    st_filtered_station = st.select(station=sta)
    print("-------------------------------------------------------")
    print("-------------------------------------------------------")
    print("Station name: ", sta)
    print("Number of traces in station: ", len(st_filtered_station))

    stationChannels(st_filtered_station)

    for cha in chan_list:
        st_filtered_channel = st_filtered_station.select(channel=cha)

        if len(st_filtered_channel) == 3:
            net = st_filtered_channel[0].stats.network
            start = [st_filtered_channel[0].stats.starttime, st_filtered_channel[1].stats.starttime,
                     st_filtered_channel[2].stats.starttime]
            start_dif = [abs(start[0] - start[1]), abs(start[1] - start[2]), abs(start[0] - start[2])]
            delta = [st[0].stats.delta, st[1].stats.delta, st[2].stats.delta]

            if (st_filtered_channel[0].data.size, st_filtered_channel[1].data.size) == (
                    st_filtered_channel[1].data.size, st_filtered_channel[2].data.size):
                samp_len = True
            else:
                samp_len = False

            if max(start_dif) < max(delta):
                if samp_len:
                    set_count += 1
                    rows = st_filtered_channel[0].data.size
                    data_log = np.empty((rows, 0), dtype='float32')
                    for tr2 in st_filtered_channel:
                        trace_data = np.reshape(tr2.data, (-1, 1))
                        data_log = np.append(data_log, trace_data, 1)

                    # filename format: network_station_channel(?)_samples.npz
                    stream_name = net + '_' + sta + '_' + cha[:-1] + '_' + str(np.size(data_log, 0)) + '.npz'
                    np.savez(os.path.join(npz_save_path, stream_name), data=data_log)
                    print("Full set of E,N,Z found for channel [{}]. Writing to ".format(cha))
                    print(os.path.join(npz_save_path, stream_name))
                else:
                    print("Start-times in channel [{}] matched but did not contain same number of samples. "
                          "Skipping...".format(cha))
                    print(st_filtered_channel)
                    len_count += 1
                    len_sta.append(sta)
            elif max(start_dif) >= max(delta):
                print("Start-times in channel [{}] don't match. Skipping...".format(cha))
                print(st_filtered_channel)
                len_count += 1
                len_sta.append(sta)
            else:
                print("Unknown false condition in channel [{}]. Skipping...".format(cha))
        elif 0 <= len(st_filtered_channel) < 3:
            print("Channel [{}] in station {} did not contain a full set of E,N,Z. Skipping...".format(cha, sta))
            incomplete_count += 1
        elif len(st_filtered_channel) > 3:
            print("More than three traces in channel [{}]. Check for duplicates. Skipping...".format(cha))
            print(st_filtered_channel.__str__(extended=True))
            overfull_count += 1
            overfull_sta.append(sta)
        else:
            print("Unknown error finding number of channels. Skipping...")

    total += set_count
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Total number of files written for station: ", set_count)

print("-------------------------------------------------------")
print("-------------------------------------------------------")
print("<<< Process finished >>>")
print("Number of traces/files in dir: ", len(st))
print("Channel filter: ", chan_list)
print("Total number of complete E,N,Z sets written: {} ({} traces)".format(total, total * 3))
print("Number of instruments with incomplete E,N,Z channels: ", incomplete_count)
print("Number of instruments with different sample lengths or start-times: ", len_count)
print("Stations: ", len_sta)
print("Number of instruments with >3 channels: ", overfull_count)
print("Stations: ", overfull_sta)

filepath = '/Users/Lenni/Downloads/NZ_DUWZ_EH_27001.npz'


def npzReader(path):
    npz_file = np.load(path)
    print("Contents: ", npz_file.files)
    data = npz_file['data']
    print("Shape: ", data.shape)
    return data


def csvWriter(source,destination):
    files = []
    for file in os.listdir(source):
        if file.endswith(".npz"):
            files.append(file)

    with open(os.path.join(destination, 'waveform.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['fname'])
        for file in files:
            writer.writerow([file])
    print("waveform.csv written to {}".format(destination))


csvWriter(npz_save_path,dataset_path)
# npzReader(filepath)

#conda activate venv
#cd /Users/Lenni/Documents/PycharmProjects/Kaikoura
#python PhaseNet/run.py --mode=pred --model_dir=PhaseNet/model/190703-214543 --data_dir=Dataset/NPZ --data_list=Dataset/waveform.csv --output_dir=output --plot_figure --save_result --batch_size=20 --input_length=27001