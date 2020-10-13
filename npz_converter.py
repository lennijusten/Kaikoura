# npz file writer
# Author: Lennart Justen
# Last revision: 6/5/20

# Description: This script takes a directory path or a directory of event directories and converts the contents into
# a three channel .npz file. Only complete sets of E,N,Z channels from a single instrument with the same number of
# samples are written. The csvWriter then writes a PhaseNet formatted csv file based on the NPZ folder contents

# Documentation
# https://numpy.org/doc/stable/reference/generated/numpy.savez.html
# https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.select.html#obspy.core.stream.Stream.select

import obspy
import numpy as np
import os
import csv
import pandas as pd
import shlex

# FILTER PARAMS
filter_method = "bandpass"
freq_min = 3.0
freq_max = 18.0

# source can be a single event folder or a folder of event folders
sac_source = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/'

chan_list = ['EH?', 'BH?']  # add channels in three letter format. Script will look for all E,N,Z channels
directions = ['N', 'E', 'Z']

tbegin = -30  # starttime is 30 seconds prior to origin of earthquake
tend = 100  # end time is 240 seconds after origin of earthquake
dt = 0.01
n_samp = int((tend - tbegin) / dt + 1)

npz_save_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset/NPZ'
dataset_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Dataset'
output_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/PhaseNet/output'
arrival_path = '/Users/Lenni/Documents/PycharmProjects/Kaikoura/Events/Arrival.pickle'

headers = ['network', 'event_id', 'station', 'channel', 'samples', 'delta', 'start', 'end', 'P_residual', 'P_time',
           'P_phasenet', 'tp_prob', 'itp', 'S_residual', 'S_time', 'S_phasenet', 'ts_prob', 'its', 'fname']

# METHODS:
# earliest -- Uses PhaseNet's earliest pick as the arrival pick
# max_prob -- Uses the PhaseNet pick with the highest probability as the arrival pick
method = 'earliest'


# Write current csv file from NPZ folder
def csvWriter(source, destination):
    files = []
    try:
        for file in os.listdir(source):
            if file.endswith(".npz"):
                files.append(file)

        with open(os.path.join(destination, 'waveform.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['fname'])
            for file in files:
                writer.writerow([file])
        print("waveform.csv written to {}".format(destination))
    except:
        print("Error writing waveform.csv. Check that directory exists.")


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


# npzReader for single npz file
def npzReader(path):
    npz_file = np.load(path)
    print("Contents: ", npz_file.files)
    data = npz_file['data']
    print("Shape: ", data.shape)
    return data


try:
    os.makedirs(npz_save_path)
    print("Directory ", npz_save_path, " Created ")
except FileExistsError:
    print("Directory ", npz_save_path, " already exists.")
    print("Writing files to existing folder...")

events = []
if os.path.basename(os.path.normpath(sac_source)) != 'Events':  # todo: fix
    events.append(os.path.basename(os.path.normpath(sac_source)))
    if len(events) == 1 and events[0] == os.path.basename(os.path.normpath(sac_source)):
        print("1 event {} found in {}".format(events, sac_source))
        single = True
else:
    single = False
    for event in os.listdir(sac_source):
        if os.path.isdir(os.path.join(sac_source, event)):
            events.append(event)
    print("{} events found in {}".format(len(events), sac_source))
    print(events)

trace_count = 0
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


run = 0
row_list = []
samples = []
for event in events:
    if single:
        try:
            run = 1
            st = obspy.read(os.path.join(sac_source, '*.SAC'))
        except Exception:
            run = 0
            print("Event {} folder is empty. Skipping...".format(event))
            pass
    else:
        try:
            st = obspy.read(os.path.join(sac_source, event, '*.SAC'))
            run = 1
        except Exception:
            run = 0
            print("Event {} folder is empty. Skipping...".format(event))
            pass

    if run == 1:
        trace_count += len(st)

        print("\n\nFetching traces from event {}...".format(event))
        print("Found {} traces. Reading... \n\n".format(len(st)))

        station = []
        for tr in st:
            station.append(tr.stats.station)
        unique_station = list(set(station))
        print("Number of unique stations: ", len(unique_station))

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
                            st_filtered_channel[1].data.size, st_filtered_channel[2].data.size) and \
                            st_filtered_channel[0].data.size == n_samp:
                        samp_len = True
                    else:
                        samp_len = False

                    if max(start_dif) < max(delta):
                        if samp_len:
                            set_count += 1
                            rows = st_filtered_channel[0].data.size
                            data_log = np.empty((rows, 0), dtype='float32')

                            # ------- FILTER METHOD ---------- #
                            st_final = st_filtered_channel.filter(filter_method, freqmin=freq_min, freqmax=freq_max)
                            filter_descrip = [filter_method, freq_min, freq_max]

                            for tr2 in st_final:
                                trace_data = np.reshape(tr2.data, (-1, 1))
                                data_log = np.append(data_log, trace_data, 1)

                            samples.append(np.size(data_log, 0))
                            # filename format: network_station_channel(?)_samples.npz
                            stream_name = net + '_' + sta + '_' + cha[:-1] + '_' + event + '_' + str(
                                np.size(data_log, 0)) + '.npz'
                            np.savez(os.path.join(npz_save_path, stream_name), data=data_log)
                            row_list.append(
                                [net, event, sta, cha, np.size(data_log, 0), tr2.stats.delta, tr2.stats.starttime,
                                 tr2.stats.endtime, filter_descrip, stream_name])

                            print("Full set of E,N,Z found for channel [{}]. Writing to ".format(cha))
                            print(os.path.join(npz_save_path, stream_name))
                        elif not samp_len:
                            print("Start-times in channel [{}] matched but did not contain same number of samples. "
                                  "Skipping...".format(cha))
                            print(st_filtered_channel)
                            len_count += 1
                            len_sta.append(event + ': ' + sta)
                        else:
                            print("Unknown error matching E,N,Z lengths for channel {}. Skipping...".format(cha))
                            print(st_filtered_channel)
                            len_count += 1
                            len_sta.append(event + ': ' + sta)
                    elif max(start_dif) >= max(delta):
                        print("Start-times in channel [{}] don't match. Skipping...".format(cha))
                        print(st_filtered_channel)
                        len_count += 1
                        len_sta.append(event + ': ' + sta)
                    else:
                        print("Unknown false condition in channel [{}]. Skipping...".format(cha))
                elif 0 <= len(st_filtered_channel) < 3:
                    print(
                        "Channel [{}] in station {} did not contain a full set of E,N,Z. Skipping...".format(cha, sta))
                    incomplete_count += 1
                elif len(st_filtered_channel) > 3:
                    print("More than three traces in channel [{}]. Check for duplicates. Skipping...".format(cha))
                    print(st_filtered_channel.__str__(extended=True))
                    overfull_count += 1
                    overfull_sta.append(event + ': ' + sta)
                else:
                    print("Unknown error finding number of channels in {}. Skipping...".format(cha))

            total += set_count
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Total number of files written for station: ", set_count)

df = pd.DataFrame(row_list, columns=['network', 'event_id', 'station', 'channel', 'samples', 'delta', 'start', 'end',
                                     'filter_method', 'fname'])

print("-------------------------------------------------------")
print("-------------------------------------------------------")
print("<<< Process finished >>>\n")

if not len(set(samples)) == 1:
    print("*** WARNING: Not all samples are the same length. Check output folder. ***")

print("Channel filter: ", chan_list)
print("Number of traces/files in dir: ", trace_count)
print("Total number of complete E,N,Z sets written: {} ({} traces)\n".format(total, total * 3))
print("Number of instruments with incomplete E,N,Z channels: ", incomplete_count)
print("\nNumber of instruments with different sample lengths or start-times: ", len_count)
print("Stations: ")
print(*len_sta, sep="\n")
print("\nNumber of instruments with >3 channels: ", overfull_count)
print("Stations: ")
print(*overfull_sta, sep="\n")

df.to_pickle(os.path.join(dataset_path, "data_log.pickle"))
print("data_log.pickle written to ", dataset_path)
df.to_csv(os.path.join(dataset_path, "data_log.csv"), index=False)
print("data_log.csv written to ", dataset_path)

csvWriter(npz_save_path, dataset_path)
# df2, p_res, s_res, p_prob, s_prob, p_empty, s_empty = csvSync(dataset_path, output_path, arrival_path, headers, method)


# pssr = np.nansum([i ** 2 for i in p_res])
# sssr = np.nansum([i ** 2 for i in s_res])
# print("P-SSR = ", pssr)
# print("S-SSR - ", sssr)

# conda activate venv
# cd /Users/Lenni/Documents/PycharmProjects/Kaikoura
# python PhaseNet/run.py --mode=pred --model_dir=PhaseNet/model/190703-214543 --data_dir=Dataset/NPZ --tp_prob=0.05 --ts_prob=0.05 --data_list=Dataset/waveform.csv --output_dir=PhaseNet/output --plot_figure --save_result --batch_size=30 --input_length=27001

# python PhaseNet/run.py --mode=pred --model_dir=PhaseNet/model/190703-214543 --data_dir=Dataset/NPZ --tp_prob=0.05 --ts_prob=0.05 --data_list=Dataset/waveform.csv --output_dir=PhaseNet/output --save_result --batch_size=50 --input_length=13001
