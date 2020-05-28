#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:42:20 2020

@author: benjaminheath
"""

# For_Lenni.py
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import read_inventory
import pandas as pd
import QUAKEML
from pathlib import Path
import numpy as np
import os

# %% parameters we can set#########################################################

### Saving info
directorytosaveto = '/Users/Lenni/Desktop/Kaikoura'  ### we will save waveforms as well as various pandas structures
# NOTE: python does not like ~s for paths so you have to be super explicit
saveformat = 'SAC'  # also MSEED, these are common data formats. The machine learning code will tell you which format to use

### Earthquake parameters
starttime = "2017-05-13 00:00:00.000"  # start date of catalog event search
endtime = "2017-05-15 00:00:00"  # end date of catalog event search

latitudeRange = [-37, -45]  # latitude range for events
longitudeRange = [166, 180]  # longitude range for events

mindepth = [0]  # minimum depth of earthquake to search for
maxdepth = [60]  # maximum depth of earthquake to search for

minmagnitude = 3  # minimum magnitude of earthquake to search for
maxmagnitude = 10  # maximum magnitude of earthquake to search for

### station parameters
station_network = 'NZ'  # station network (NZ = New Zealand, other networks have different two letter codes)
chan = '*Z'  # channels are labeled ??[Z/N/E]. Z means vertical (N = North, E = East). Usually the stations we use are either just vertical or all 3 components
# by selecting just the vertical components we can get all the data we need (station name, location and elevation)


################################################################################


# %% initialize client (This is where the data is stored on line)
client = FDSN_Client("GEONET")

# %% grab metadata (EQ and Station information)


### get earthquake info
print('Getting Events')
cat = client.get_events(starttime=starttime, endtime=endtime, minlatitude=latitudeRange[1],
                        maxlatitude=latitudeRange[0], \
                        minlongitude=longitudeRange[0], maxlongitude=longitudeRange[1], minmagnitude=minmagnitude, \
                        mindepth=mindepth[0], maxdepth=maxdepth[0])  # run client to grab events within ranges

#       _=cat.plot(projection="local")# plot simple view of events

### get station info
print('Getting Stations')
stanet = client.get_stations(network='NZ', channel='*Z', level="channel", starttime=starttime, \
                             endtime=endtime)  # grab station information

# %% process
# cat and sta are not as easy as we would like to use. so we convert them to easy-to-use pandas dataframes

print('In quaekeml')
quakeml = QUAKEML.QuakeML(cat)

print('to Events')
Events = quakeml.to_Event()  ### this creates a pandas array that has event_id, event location and event depth

print('to Arrival')
Arrival = quakeml.to_Arrival()  #### this creates an arrival array with station, time (absolute), error, phase and event_id
Arrival = Arrival.drop_duplicates(subset=['event_id', 'station'])

### put stations into easy to use structures
stations = {'station_latitude': [], 'station_longitude': [], 'station_elevation': [], 'station_name': [],
            'station_network': [], 'station_channel': []}

for inet, network in enumerate(stanet):
    net = network.code
    for station in network:
        station_name = station.code
        ### append
        stations['station_longitude'].append(station.longitude)
        stations['station_latitude'].append(station.latitude)
        stations['station_elevation'].append(int(station.elevation))
        stations['station_name'].append(station.code)
        stations['station_network'].append(net)
        stations['station_channel'].append(station.channels[0].code[
                                           0:2] + '*')  # here I am using ??[ZNE] and actually concerned with the ?? part (technical reasons/rarely used )

Stations = pd.DataFrame(stations)

# %% save pandas files for various structrues
Arrival.to_pickle(directorytosaveto + '/Arrival.pickle')
Events.to_pickle(directorytosaveto + '/Event.pickle')
Stations.to_pickle(directorytosaveto + '/Station.pickle')

# %% now we get waveforms

for iarr,_ in enumerate(Arrival['event_id']):

    station = Arrival['station'].values[iarr]  # grab station name
    network = Arrival['network'].values[iarr]  # grab network
    phase = Arrival['phase'].values[iarr]  # grab phase
    event_id = Arrival['event_id'].values[iarr]  # grab event_id

    ievt = np.where(Events['event_id'].values == Arrival['event_id'].values[iarr])[0][0]
    otime = Events['event_origin_time'].values[ievt]  # grab origin time (in Event info)

    # things we need for grabbing data
    chan = 'ZNE'  # channel information
    location = '*'  # location information
    tbegin = otime - 30  # starttime is 30 seconds prior to origin of earthquake
    tend = otime + 240  # end time is 240 seconds after origin of earthquake

    # make event_id directory to save file in
    Path(directorytosaveto + '/' + event_id).mkdir(parents=False, exist_ok=True)

    # get waveforms
    print('Getting Arrival ' + str(iarr))
    for cha in chan:
        try:
            st = client.get_waveforms(network, station, location, '*' + cha, tbegin, tend)
            for tr in st:
                filename = network + '_' + station + '_' + tr.stats.channel + '.' + saveformat
                local_path = os.path.join(directorytosaveto, event_id, filename)
                if not os.path.exists(local_path):
                    print('Writing ' + local_path)
                    tr.write(local_path, format=saveformat)
                else:
                    print('File already exists: ', local_path)
        except:
            print('Unable to get waveforms for ' + event_id + ' ' + station)

##### PSEUDO CODE FOR PLOTTING WAVEFORMS
# import matplotlib.pyplot as plt
# count = 0
# st = obspy.read(filename)
# for tr in st:
#       tr.filter('bandpass',freqmin=2,freqmax = 10) # frequencies to filter (in hz), usually between 1-20 Hz
#       plt.plot(tr.times(),tr.data/max(tr.data)+count)
#       count += 1
