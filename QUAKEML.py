#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:33:47 2020

@author: benjaminheath
"""

class QuakeML:
        
        def __init__(self,quakemlcat):
                
                ## initialize fields
                quakeml ={'event_id':[],'event_latitude':[],'event_longitude':[],'event_depth':[],'event_origin_time':[],'event_magnitude':[],\
                          'station':[],'time':[],'phase':[],'error':[],'arrival_latitude':[],'channel':[],\
                                  'arrival_longitude':[],'method':[],'event_sx':[],'event_sy':[],'event_sz':[],'network':[]}
                
        
                for event in quakemlcat:
                        #print(event.preferred_origin())
                        if event.preferred_origin() is None:
                                print('QUAKEML: Skipping event '+str(event.resource_id).split('/')[-1]+' ... no preferred origin')
                                continue
                        for key in quakeml.keys():
                                value = (self._get_Value(event,key))
                                for val in value:
                                        quakeml[key].append(val)
                
                                        
                self.quakemlcatalog = quakeml
                
        def to_Arrival(self):
                
                import pandas as pd
                
                arrival_keys = ['event_id','station','phase','time','error','method','network','channel']
                arrival = self.quakemlcatalog.copy()
                
                for key in self.quakemlcatalog.keys():
                        if key not in arrival_keys:
                                del arrival[key]
                
                Arrival = pd.DataFrame(arrival)
                
                
                return Arrival
                
                
        def to_Event(self):
                
                import pandas as pd
                
                event_keys= ['event_id','event_latitude','event_longitude','event_depth','event_origin_time','event_magnitude']
                events = self.quakemlcatalog.copy()
                for key in self.quakemlcatalog.keys():
                        if key not in event_keys:
                                del events[key]
                Events = pd.DataFrame(events)
                Events = Events.drop_duplicates(subset = ['event_id'], keep = 'first')
                Events = Events.reset_index(drop = True)
                
                return Events

        def _get_Value(self,event, field):
                ''' Gets a certain field from quakeml file
                
                valid fields are:
                        event_id
                        event_latitude
                        event_longitude
                        event_depth
                        event_origin_time
                        event_magnitude
                        arrival_station
                        arrival_time
                        arrival_phase
                        arrival_error
                        arrival_type

                '''
                from numpy import nan as nan                
                len2match = len(event.picks)
                found = 0
                if field == 'event_id':
                        value =  str(event.resource_id).split('/')[-1]
                        found= 1
                if field == 'event_origin_time':
                        value =  event.preferred_origin().time
                        found= 1
                if field == 'event_latitude':
                        value =  event.preferred_origin().latitude
                        found= 1
                if field == 'event_longitude':
                        found= 1
                        value =  event.preferred_origin().longitude
                if field == 'event_depth':
                        value = event.preferred_origin().depth/1000
                        found= 1
                if field == 'time':
                        value = [pick.time for pick in event.picks]
                        found= 1
                if field == 'error':
                        value = [pick.time_errors.uncertainty if pick.time_errors.uncertainty is  not None else -100 for pick in event.picks]
                        for ii,_ in enumerate(value):
                                if value[ii] == -100:
                                        value[ii] =nan
                                                                                
                        found= 1
                if field == 'method':
                        value = [pick.evaluation_mode for pick in event.picks]
                        found= 1
                if field == 'station':
                        value = [pick.waveform_id.station_code for pick in event.picks]
                        found= 1
                if field == 'network':
                        value = [pick.waveform_id.network_code for pick in event.picks]
                        found= 1
                if field == 'phase':
                        value = [pick.phase_hint for pick in event.picks]
                        found= 1
                if field == 'channel':
                        value = [pick.waveform_id.channel_code for pick in event.picks]
                        found = 1
                if field == 'event_magnitude':
                        value = event.preferred_magnitude().mag
                        found = 1
                if found == 0:
                        value = nan
                        
                value = self._match_length(value,len2match)
                return value 
                
        def _match_length(self,value,len2match):
                
                if type(value) is not list:
                        vv = list()
                        vv.append(value)
                        value = vv
                
                if len(value) != len2match:
                        if len(value)!= 1:
                                raise ValueError('This should either be 1 or len2match')
                        value_update = []
                        for i in range(0,len2match):
                                value_update.append(value[0])
                        value = value_update
                
                return value
                        
                
                
                
                
                
                
                
                
                
                