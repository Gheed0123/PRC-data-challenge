# -*- coding: utf-8 -*-
"""
Created by Herpa Derp
"""

import pandas as pd
import numpy as np
import country_converter as coco
import pathlib

from models import model_TOW
from Timezones import get_tz

# %%load and init
#
folder = pathlib.Path.cwd().parents[2] / "Data/ATOW"
df = pd.read_csv(folder / 'pq_eval.csv')

cset = pd.read_csv(folder / r"challenge_set.csv")
sset = pd.read_csv(folder / r"final_submission_set.csv")

df = pd.concat([df, cset, sset]).drop_duplicates(keep='first', subset='flight_id').reset_index(drop=True)

ft2m = 0.3048
kt2ms = 0.514444
lbs2kg = 0.45359237

apdata = pd.read_excel(folder / r"FAA-Aircraft-Char-DB-AC-150-5300-13B-App-2023-09-07.xlsx")
other_data = pd.read_csv(folder / r"other_data.txt")

other_data.loc[other_data.index[:-1], 'MTOW_lbs'] = (other_data.MTOW_lbs[:-1].astype(float)*lbs2kg).round()

# ICAO codes not in manual data (for passenger amount, MTOW2 etc)
print('ICAO codes not in manual data (for passenger amount, MTOW2 etc)')
print(set(df[~df.aircraft_type.isin(other_data.ICAO_Code)].aircraft_type))

df = df.merge(other_data, left_on='aircraft_type', right_on='ICAO_Code', how='inner')
df.MTOW_lbs = df.MTOW_lbs.astype(float).round()
df.passengers_max = df.passengers_max.fillna(df.passengers_typical).astype(int)

# %% fix all the needed data for the model

# mtow data add
apdata['MTOW'] = apdata.MTOW_lb*lbs2kg
aplanes = [apdata[apdata.ICAO_Code == adata] if (adata in list(apdata.ICAO_Code)) else apdata[apdata.ICAO_Code ==
                                                                                              'A320']*0 for adata in list(set(df.aircraft_type))]
aplanes = pd.concat(aplanes)
MTOW = aplanes.MTOW.astype(float).round()
MTOWS = dict(zip(aplanes.ICAO_Code, aplanes.MTOW))
MTOWS = [MTOWS[c] for c in df.aircraft_type]
df['MTOW'] = MTOWS
df.MTOW = df.MTOW.astype(float).round()

# check for possible new generation planes
heavierplanes = list(set(df[df.MTOW <= df.tow-1000].aircraft_type))
heavierflights = df[df.MTOW <= df.tow-1000]

# a cessna weighing >5x MTOW ╭(-᷅_-᷄ 〝)╮
cessna = df[df.flight_id == 254959338]
df = df[df.flight_id != 254959338]

# mainly plane type CRJ9
# Likely the FAA list does not include (or I fail to include) the fact that these planes have different MTOW in variants of the planes
# manually fix this (increase mtow to secondary level defined in other data per airline, )
# might have issues in other planes/airlines, but unable to fix that since we dont know more than ICAO code
heavierairline = df[(df.aircraft_type.isin(heavierplanes)) & (df.airline.isin(heavierflights.airline))]

# timezones
df.date = pd.to_datetime(df.date)
df = get_tz(df)
df.arrival_time = [pd.Timestamp(arr, tzinfo=df.tz_ades[i]) for i, arr in enumerate(df.arrival_time)]
df.actual_offblock_time = [pd.Timestamp(obt, tzinfo=df.tz_adep[i]) for i, obt in enumerate(df.actual_offblock_time)]

# jetlag factor
df['delta_tz'] = [arr.utcoffset()-df.actual_offblock_time[i].utcoffset() for i, arr in enumerate(df.arrival_time)]
df['jetlag'] = ((df.delta_tz.dt.total_seconds()/3600) > 6)*1
df['delta_tz'] = df.delta_tz.dt.total_seconds()/3600
# spring=(df.date<pd.Timestamp(year=2022,month=6,day=21))&(df.date>pd.Timestamp(year=2022,month=3,day=21))
# IATA schedule
df['IATA'] = ((df.date < pd.Timestamp(year=2022, month=10, day=30)) & (df.date > pd.Timestamp(year=2022, month=3, day=27)))*1
# weeks (holiday seasons?)
df['week'] = df.date.dt.isocalendar().week.astype(object)
df['month'] = df.date.dt.month
df['day'] = df.date.dt.day.astype(object)
df['weekday'] = df.date.dt.weekday.astype(object)
df['season'] = (df.date.dt.month % 12 // 3 + 1).astype(object)

# holiday depends on country I guess... maybe for later
# https://hub.worldpop.org/doi/10.5258/SOTON/WP00693
hooray = pd.read_csv(folder / "month_public_and_school_holidays_2010_2019.csv")
hooray = hooray[hooray.Year == 2019]
hooray['holidays'] = hooray.holiday+hooray.hl_sch
hooray['ISO2'] = coco.convert(names=hooray.ISO3, to='ISO2')
hooray = hooray[['ISO2', 'Month', 'holidays']]
hooray.columns = ['country_code_adep', 'month', 'holidays']
df = df.merge(hooray, how='left')
df['holidays'] = df['holidays'].fillna(0)
df['month'] = df.date.dt.month.astype(object)

# workweek assume 5 days!! except holidays...
df['weekend'] = (df.weekday > 4)*1

# morning/afternoon/evening/night start/end categorical
# shifter for moving towards and aways from airport, this has near zero effect on model it seems

shift_end = 1
shift_start = 2

h = df.actual_offblock_time.apply(lambda x: x.hour)
h -= shift_start
h.loc[h < 0] = 24+h.loc[h < 0]

startblock = h.copy(deep=True).astype(object)
startblock.loc[h < 6] = 'night'
startblock.loc[(h > 5) & (h < 12)] = 'morning'
startblock.loc[(h > 11) & (h < 18)] = 'afternoon'
startblock.loc[h > 17] = 'evening'

h = df.arrival_time.apply(lambda x: x.hour)
h += shift_end
h.loc[h > 23] = h.loc[h > 23]-24

endblock = h.copy(deep=True).astype(object)
endblock.loc[h < 6] = 'night'
endblock.loc[(h > 5) & (h < 12)] = 'morning'
endblock.loc[(h > 11) & (h < 18)] = 'afternoon'
endblock.loc[h > 17] = 'evening'
# maybe prepare 2-3 hours extra for start flights, but people might just get hotel or stay at airport?
# and 0-1h for end flight

df['startblock'] = startblock
df['endblock'] = endblock

# flight duration
df['flight_duration'] = (df.arrival_time.apply(pd.Timestamp)-df.actual_offblock_time.apply(pd.Timestamp)).apply(pd.Timedelta.total_seconds)

# amount of flights on same day, might give indication for mroe bookings -> more tow if airlines use this to dispatch more (of the) planes
# and if the selected sets are representative
df['daily_flights'] = df.date.groupby(df.date).transform('size')

# flightpath (heen/terug pad klax-egll en egll-klax zelfde, of net anders?)
df['flightpathU'] = df.adep+df.ades
df['flightpath'] = [''.join(sorted(a)) for a in df.adep+df.ades]
df['flightpath_Airline'] = df['flightpathU']+df.airline+df.MTOW.astype(str)

# fix domestic
df['domestic'] = (df.country_code_adep == df.country_code_ades)*1

# taxiout time
df['taxiout_time2'] = df.taxiout_time.astype(object).apply(pd.Timedelta).apply(lambda x: x.total_seconds())/60

# check tow stuff
df['groupsz'] = -1
df['group'] = df.groupby(['flightpath_Airline']).ngroup()
df.loc[df.tow > 0, 'groupsz'] = df[df.tow > 0].groupby(['flightpath_Airline']).transform('size')

#
df['flightpath_Airline_MTOW'] = df.flightpathU+df.MTOW.astype(str)
df = df.sort_values(['MTOW', 'flightpath_Airline_MTOW', 'tow']).reset_index(drop=True)

df['factor_back'] = 1

# %%booleans to int, because apparently cannot check .corr without it
# also make some vars bool, might make for better fits
# too late to investigate humidity, assumed 0.009 RH == 90% humid air and that is put in model??
df.good = df.good*1
df.holidays = (df.holidays >= 4)*1
# df['possible_freezing_start'] = (df.temp_start <= 278)*1
# df['freezing_start'] = (df.temp_start <= 273)*1
# df['very_humid'] = (df.humid_start > .009)*1
# df['high_alt_takeoff'] = ((df.alt_start > 9000) & (df.alt_start < 20000))*1
# df['possible_snowy_']
df.passengers_typical = df.passengers_typical.astype(float)

# weight indexing more than wtc
splits = [10e3, 25e3, 40e3, 65e3, 100e3, 175e3, 275e3, 999e9]
df['weight_class'] = ''
for split in splits[::-1]:
    df.loc[df.MTOW < split, 'weight_class'] = split

# other stuff
lug = 17  # checked in luggage weight
mpw = 84+lug  # meanpassengerweight including carryon, luggage potentially depends on flight distance?? also on destination (holidayY?) etc etc see study
# just take something for now , study says 17kg

# % to fill passenger planes. taking average for year... should take domestic vs not.. modelling regression stuff see paper for variables
load_factor_year = 82.5/100

# fix too small flown distance (data missing??) by making flown distance 1 leading in those cases (fd1>fd2)
# check factor
df.loc[(df.flown_distance*1000) > df.flown_distance2, 'flown_distance2'] = df.loc[(df.flown_distance*1000) > df.flown_distance2, 'flown_distance']

# etops fix for ordinal/int, rought estimate of extra fuel w the flight duration taken into account
# shorter flights might need more fuel anyways for alternate airports
# non direct route (result high) can also require more than etops route (north atlantic and hawaii?) for same start and endpoints

# %%last minute stuff
# for c in df.columns:
#     if ((df[c].dtype in [int, float, np.int64, np.float64]) & (c != 'MTOW') & (c != 'tow')):
#         df[c+'_delta'] = df.groupby('MTOW')[c].transform('mean')-df[c]

# for c in df.columns:
#     if ((df[c].dtype in [int, float, np.int64, np.float64]) & (c != 'MTOW') & (c != 'tow') & ('_delta' not in c[-7:])):
#         df[c+'_delta2'] = np.sqrt((df.groupby(['flightpathU'])[c].transform('mean')-df[c])/df.groupby(['flightpathU'])[c].transform('mean')).fillna(0)

# for c in df.columns:
#     if ((df[c].dtype in [int, float, np.int64, np.float64]) & (c != 'MTOW') & (c != 'tow') & ('_delta' not in c[-7:])):
#         df[c+'_delta2'] = (df.groupby(['MTOW', 'flightpathU'])[c].transform('mean')-df[c])

for c in ['flown_distance', 'flight_duration']:
    if ((df[c].dtype in [int, float, np.int64, np.float64]) & (c != 'MTOW') & (c != 'tow') & ('_delta' not in c[-7:])):
        df[c+'_delta2'] = (df.groupby(['MTOW', 'flightpathU'])[c].transform('mean')-df[c])

# for c in df.columns:
#     if ((df[c].dtype in [int, float, np.int64, np.float64]) & (c != 'MTOW') & (c != 'tow') & ('_delta' not in c[-7:])):
#         df[c+'_delta3'] = (df.groupby(['MTOW', 'airline'])[c].transform('mean')-df[c])

# %%check if previous and next flight can be found
df['previous'] = df.sort_values(['callsign', 'actual_offblock_time']).groupby('callsign').tow.shift()
df['next'] = df.sort_values(['callsign', 'actual_offblock_time']).groupby('callsign').tow.shift(-1)

# %%unique airplanes per airline
df_zeroes = df[['airline', 'callsign']].copy()
df_zeroes['zeroes'] = 0
q = df[['airline', 'callsign']].groupby(['airline', 'callsign']).nth(0).reset_index()
df_zeroes.zeroes[q['index']] = 1
df['callsign'] = (df_zeroes.groupby(['airline']).zeroes.cumsum()-1).astype(str)

# mean tow for pathairlineairplane
mean_TOW = df.groupby(['flightpath_Airline', 'callsign', 'MTOW']).tow.transform('mean')
mean_TOW = mean_TOW.fillna(df.groupby(['flightpath_Airline', 'MTOW']).tow.transform('mean'))
mean_TOW = mean_TOW.fillna(df.groupby(['flightpathU', 'MTOW']).tow.transform('mean'))
mean_TOW = mean_TOW.fillna(df.groupby(['flightpath', 'MTOW']).tow.transform('mean'))
mean_TOW = mean_TOW.fillna(df.groupby(['MTOW']).tow.transform('mean'))

# to add: fuel tankering stuff
# I dont have any data on this, so I made some up ;)
# TOW delta, per airport per month
# Hopefully this can capture fuel shortages or cheap fuel locations that are not captured by ades, adep or countrycode

# df['tank_here'] = df.groupby(['month', 'flightpath_Airline']).tow.transform('mean')
# df['tank_here_b'] = ((df.groupby(['month', 'flightpath_Airline']).tow.transform('mean')) > .9)*1
# df['dont_tank_here_b'] = ((df.groupby(['month', 'flightpath_Airline']).tow.transform('mean')) < -.9)*1
# df[(df.tank_here>.9) & ~df.tow.isna()].adep.value_counts()

# %%indicator for small samplesizes per  unique flightpath+airline+tow, to use other model for it.
# dont use this var for regression
groupsz = 10  # 10 flights should be fiine
c = 'flightpath_Airline'
c = 'callsign'
groupsz_full = df.groupby([c]).transform('size')
df['fullmodel'] = (groupsz_full//groupsz) < 1
# (groupsz_full//groupsz).value_counts()


# %%model me this model me that
df_model = df.copy(deep=True)

# model_TOW(df,mean_TOW=mean_TOW)

# %%other param stuff to maybe test
# sleep possibility (night and long time)
# fix final stuff model code refactoring
# mean_tow fix stuff, no model needed if it is the same.
# fix correlated categorical vars (like flightpath, flightpathu)
