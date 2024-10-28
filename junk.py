# -*- coding: utf-8 -*-
"""
@author: Herpa Derp
Random junk here
"""
import json
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rmse_manual(output, train, tow):
    if (type(output) == list):
        rmse = np.sqrt((((np.concatenate(output)-np.concatenate(train))*tow)**2).mean())
    else:
        rmse = np.sqrt((((output-train)*tow)**2).mean())
    print('rmse: ', rmse)
    return


def rmse(df, col):
    vals = (df[col]-df.tow).dropna()
    rmse = np.sqrt((vals**2).mean())
    print('rmse: ', rmse)
    return


def save_pred(pred, num, col='tow'):
    js = pathlib.Path.cwd() / [file for file in os.listdir(pathlib.Path.cwd()) if file.endswith('.json')][0]
    with open(js) as f:
        jdata = json.load(f)

    teamname = jdata['team_name']
    team_id = jdata['team_id']
    filename = teamname+'_v'+str(num)+'_'+team_id+'.csv'
    path = pathlib.Path.cwd().parents[2] / 'Data/ATOW/submissions'
    p = pred[['flight_id', col]].copy()
    p.columns = ['flight_id', 'tow']
    if (p.tow.isna().any()):
        print('found na in predictions!!')
    p['tow'] = p['tow'].round().fillna(0.0)
    p.to_csv(path / filename, index=False)
    return

# %%


def stats_error_grouping(dff, errors, df_train):
    # mostly heavy flights. mostly estimated too low
    cols = [c for c in dff.columns if type(dff[c][0]) in [int, float, str, np.int64, np.float64, bool]]
    dff = dff[cols].copy()

    if (not errors.empty):
        errorflights = dff[dff.flight_id.isin(errors.flight_id.unique())]
        errorflights = pd.merge(errorflights, errors, how='left', on=['flight_id'])

        dff = dff.merge(df_train, how='left', on='flight_id')
        q = dff[(dff.airline == errorflights.airline.value_counts().index[0]) & (dff.flightpathU == errorflights.flightpathU.value_counts().index[0])]
        q = q.sort_values(by=['MTOW', 'tow_x', 'tow_total'], axis=0)

    # looks like tangent.. also my predictions are very overfit
    dft = df_train.copy().sort_values(by=['tow', 'tow_total'], axis=0)
    for c in df_train.columns[2:]:
        plt.figure()
        plt.title('tow and '+c)
        plt.plot(dft[c].sort_values().dropna().values, label=c)
        plt.plot(dft[c].dropna().values, label=c)
        plt.plot(dft['tow'].dropna().values, label='tow')
        plt.legend()
    return

# %%


def plot_squared_errors(df, col, cutoff=1e8):
    errors = ((df[col]-df.tow)**2)
    e = errors.values.copy()
    e.sort()
    plt.plot(e)
    return df[errors > cutoff]

# %%


def adjust_vars(df, rf=False):
    df2 = df.copy(deep=True)

    df2.wind_advantage = np.sign(df2.wind_advantage)*np.sqrt(abs(df2.wind_advantage))
    df.ETOPS.replace(0, np.nan, inplace=True)

    # model tuning parameters
    df2 = df2.drop(['fullmodel',
                    # 'airline',
                    'aircraft_type',
                    # 'flown_distance2',#
                    # 'passengers_typical',
                    # 'MTOW,
                    'callsign',
                    'date',
                    'name_adep',
                    'name_ades',
                    'actual_offblock_time',
                    'arrival_time',
                    'taxiout_time',
                    # 'wtc',
                    # 'flown_distance',
                    'ICAO_Code',
                    # 'passengers_max',
                    # 'factor_back',
                    # 'adep',
                    # 'ades',
                    'tz_adep',
                    'tz_ades',
                    'country_code_ades',
                    'country_code_adep',
                    # 'taxiout_time2',
                    # 'week',
                    'day',
                    # 'month',
                    # 'weekday',
                    # 'IATA',
                    'flightpath',
                    'flightpathU',
                    'flightpath_Airline',
                    'flightpath_Airline_MTOW',
                    # 'flight_duration',
                    # 'wind_advantage',
                    'weight_class',
                    # 'mean_TOW',
                    # 'startblock',
                    # 'endblock',
                    # 'daily_flights',
                    # 'ETOPS',
                    # 'made_stop',
                    # 'temp_start',
                    # 'temp_end',
                    # 'EAS_mean',
                    # 'speed',
                    # 'fuel_factor',
                    'MTOW_lbs'
                    ], axis=1)

    othercols = list(df2.select_dtypes(include=[object]).columns)

    for col in othercols:
        df2[col] = df2[col].astype('category')

    return df2
