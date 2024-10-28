# -*- coding: utf-8 -*-
"""
@author: Herpa Derp
"""

import pandas as pd
import numpy as np
from junk import rmse, save_pred, adjust_vars, plot_squared_errors, stats_error_grouping
import matplotlib.pyplot as plt
from time import time
import pathlib
from xgboost import XGBRegressor, plot_importance

# %%


def test_XGB(train, test):
    # Initialize and train the XGBoost model.

    # 10k estimators gives good rmse, but for sure is overfit
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, alpha=0.5, max_depth=10,
                         colsample_bytree=0.9, learning_rate=0.05, enable_categorical=True)

    model.fit(train[0], train[1])
    plot_importance(model, max_num_features=15)

    y_pred = model.predict(test[0])
    return y_pred, model


def select_testset(df):
    # check 1 in 10

    submit = df[df.tow.isna()]
    dft = df[~df.tow.isna()]
    tow = dft.tow
    dft = dft.drop('tow', axis=1)
    submit = submit.drop('tow', axis=1)
    df = df.drop('tow', axis=1)

    othercols = list(dft.select_dtypes(include=[int, float, np.float64, np.int64, bool]).columns)

    for c in othercols:
        if (dft[c].isna().any()):
            print('found nan in col: ', c)
            dft[c] = dft[c].fillna(0)
        if (submit[c].isna().any()):
            print('found nan in submitset col: ', c)
            submit[c] = submit[c].fillna(0)
        df[c] = df[c].fillna(0)

    # dfr = [dft, dft]
    # towr = [tow, tow]
    # submitr = [submit, submit]

    return [dft], [tow], [submit]


def predict(model, submit, tow_max):
    if (type(model) == list):
        output = [(m.predict(submit[i].drop('flight_id', axis=1))*tow_max) for i, m in enumerate(model) if (not submit[i].empty)]
        ds = []
        i = 0
        for s in submit:
            if (s.empty):
                continue
            d = pd.DataFrame(s['flight_id'])
            d['tow'] = output[i]
            ds += [d]
            i += 1
        ds = pd.concat(ds)
        output = ds
    else:

        output = model.predict(submit)*tow_max
        # tbc
        pass
    return output


def df_model_merge(df, df_model, tow_col):
    keepcols = ['flight_id', 'tow']
    df_m = df_model[keepcols].copy()
    df_m.columns = ['flight_id', tow_col]
    df = df.merge(df_m, how='left', on='flight_id')
    return df


def make_model(dff, settings):
    a = time()
    print('making model with settings: ', settings)
    tow_max = settings[0]
    no_flightdata = settings[1]
    if (not no_flightdata):
        pass
        #dff = dff[dff.ETOPS > .1].reset_index(drop=True)

    xtrain, ytrain, submit = select_testset(dff)
    output_models = [test_XGB([xtrain[i].drop('flight_id', axis=1), ytrain[i]], [xtrain[i].drop('flight_id', axis=1), ytrain[i]])
                     for i, x in enumerate(xtrain) if (not x.empty)]

    outputs = [o[0] for o in output_models]
    model = [o[1] for o in output_models]

    idx = np.concatenate([x.flight_id.values for x in xtrain])
    output = pd.DataFrame([idx, np.concatenate(outputs)*tow_max]).T
    output.columns = ['flight_id', 'tow']

    prediction = predict(model, submit, tow_max)
    print(time()-a)
    return output, prediction
# %%


def model_TOW(df, tow_nomodel=None, mean_TOW=None):
    # df=df[:20000]
    dff = df.copy(deep=True)

    folder = pathlib.Path.cwd().parents[2] / "Data/ATOW"
    cset = pd.read_csv(folder / r"challenge_set.csv")

    tow_max = 1
    dff = adjust_vars(dff)

    # complete model

    no_flightdata = False

    settings = [tow_max, no_flightdata]
    output_test_XGB, pred_test_XGB = make_model(dff, settings)

    # a totally different model for flights with a lot of missing ads-b data
    no_flightdata = True
    settings = [tow_max, no_flightdata]

    lst = df[list(df.columns[(~df.isna().any())])+['tow', 'previous', 'next']]
    dff_noadsb = dff.copy()[dff.columns[dff.columns.isin(lst)]]

    output_XGB_noadsb, pred_XGB_noadsb = make_model(dff_noadsb, settings)

    pred_XGB_noadsb = pred_XGB_noadsb[pred_XGB_noadsb.flight_id.isin(df.flight_id[~(df.good == 1)])]
    output_XGB_noadsb = output_XGB_noadsb[output_XGB_noadsb.flight_id.isin(df.flight_id[~(df.good == 1)])]

    # %%
    df_train = df[~df.tow.isna()][['flight_id', 'tow']].copy().reset_index(drop=True)
    df_submit = df[df.tow.isna()][['flight_id', 'tow']].copy().reset_index(drop=True)

    # add all stuff together
    dfmtow = df[['flight_id']]
    dfmtow['tow'] = mean_TOW

    # df_train = df_model_merge(df_train, dfmtow, 'mean_TOW')
    # df_submit = df_model_merge(df_submit, dfmtow, 'mean_TOW')

    df_train = df_model_merge(df_train, output_test_XGB, 'test_XGB')
    df_submit = df_model_merge(df_submit, pred_test_XGB, 'test_XGB')

    df_train = df_model_merge(df_train, output_XGB_noadsb, 'tow_xgb_noadsb')
    df_submit = df_model_merge(df_submit, pred_XGB_noadsb, 'tow_xgb_noadsb')

    # total model, forward filling
    df_train['tow_total'] = df_train.iloc[:, 2:].T.ffill().T[df_train.columns[-1]]
    df_submit['tow_total'] = df_submit.iloc[:, 2:].T.ffill().T[df_submit.columns[-1]]

    # fix the ones above and too far below MTOW
    mn = 2
    df_train = df_train.merge(df[['flight_id', 'MTOW']])
    toomuchtow_t = ((df_train.tow_total > (df_train.MTOW+df_train.MTOW/50)))
    df_train.loc[toomuchtow_t, 'tow_total'] = df_train.MTOW[toomuchtow_t]
    toomuchtow_t = ((df_train.tow_total < (df_train.MTOW/mn)))
    df_train.loc[toomuchtow_t, 'tow_total'] = df_train.MTOW[toomuchtow_t]/mn
    df_train = df_train.drop('MTOW', axis=1)

    df_submit = df_submit.merge(df[['flight_id', 'MTOW']])
    toomuchtow_s = ((df_submit.tow_total > df_submit.MTOW))
    df_submit.loc[toomuchtow_s, 'tow_total'] = df_submit.MTOW[toomuchtow_s]
    toomuchtow_s = ((df_submit.tow_total < (df_submit.MTOW/mn)))
    df_submit.loc[toomuchtow_s, 'tow_total'] = df_submit.MTOW[toomuchtow_s]/mn
    df_submit = df_submit.drop('MTOW', axis=1)

    mean_tow_submit = df[['flight_id']].copy()
    mean_tow_submit['mean_tow'] = mean_TOW
    df_submit = df_submit.merge(mean_tow_submit, how='left')
    df_submit.tow_total = df_submit.tow_total.fillna(df_submit.mean_tow)
    df_submit = df_submit.drop('mean_tow', axis=1)

    print('rmse individual models, dropna')
    for c in df_train.columns[2:-1]:
        print(c)
        rmse(df_train, c)

    print('rmse total model')
    _df = df_train.iloc[:, 2:-1].ffill(axis=1)
    _df['tow'] = df_train.tow
    rmse(_df, _df.columns[-1])

    print('rmse total model & nomodel')
    rmse(df_train, 'tow_total')

    num = 21
    save_pred(df_submit, num=num, col='tow_total')
    errors = plot_squared_errors(df_train, col='tow_total')
    stats_error_grouping(df, errors, df_train)

    r1 = np.arange(0, len(df_train.tow_total)-1, len(df_train.tow_total)/len(df_submit.tow_total))
    plt.figure()
    plt.plot(r1, df_submit.tow_total)
    plt.plot(df_train.index, df_train.tow_total)
    plt.plot(r1, df_submit.tow_total.sort_values())
    plt.plot(df_train.index, df_train.tow_total.sort_values())
    plt.figure()
    plt.plot(df_train.index, abs(df_train.tow_total.sort_values()-df_train.sort_values('tow_total').tow))

    return  # [model1,model2],pred


# model_TOW(df, mean_TOW=mean_TOW)
