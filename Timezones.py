# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:35:49 2024

@author: Herpa Derp
"""

import datetime
from timezonefinder import TimezoneFinder
import pathlib
import pandas as pd
#from pytz import timezone, utc
from datetime import timezone
from zoneinfo import ZoneInfo

#inspired from https://gist.github.com/mattjohnsonpint/6d219c48697c550c2476
def get_tz(df):
    _file=pathlib.Path.cwd().parents[2] / "Data/ATOW/airports.csv"
    airports=pd.read_csv(_file)
    
    tf = TimezoneFinder()
    lon=airports.longitude_deg
    lat=airports.latitude_deg
    
    tz=[ZoneInfo(tf.timezone_at(lng=lon[i],lat=lat[i])) for i,_ in enumerate(lon)]
    airports['tz']=tz
    airports=airports[['ident','tz']]
    
    df=df.merge(airports,left_on='ades',right_on='ident',how='left')
    df['tz_ades']=df.tz.fillna(ZoneInfo('UTC'))
    df=df.drop('tz',axis=1)
    df=df.drop('ident',axis=1)
    
    df=df.merge(airports,left_on='adep',right_on='ident',how='left')
    df['tz_adep']=df.tz.fillna(ZoneInfo('UTC'))
    df=df.drop('tz',axis=1)
    df=df.drop('ident',axis=1)
    
    return df