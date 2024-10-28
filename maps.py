# -*- coding: utf-8 -*-
"""
Created by Herpa Derp
"""
import pandas as pd
import geopandas as gpd
import pathlib
import country_converter as coco
from great_circle_vectorized import draw_buffer_around_points
import numpy as np


def get_etops():
    print('getting etops data')
    _file = pathlib.Path.cwd().parents[2] / "Data/ATOW/airports.csv"

    conflict_zones = pathlib.Path.cwd().parents[2] / r"Data/ATOW/conflict_zone_information_bulletin_czibs.csv"
    land_borders = pathlib.Path.cwd().parents[2] / r"Data\ATOW\world-administrative-boundaries.geojson"
    airports = pd.read_csv(_file)
    ldf = gpd.read_file(land_borders)

    # get etops ready, eez not taken into account, only country borders
    ldf = ldf[['iso_3166_1_alpha_2_codes', 'geometry']]
    ldf = ldf[~ldf.isna().any(axis=1)]
    ldf.columns = ['country', 'geometry']
    # add these zones to ETOPS stuff
    # Assume airports in these zones are out of use and the planes thus fly around to not take on much extra fuel incase of having to deviate
    # if warning and they go through, they might not be able to land, so they gtfo.
    # some inconsistencies Im not willing to fix (e.g. south of country X is blocked, but with this simple script the whole country is blocked)
    df_cz = pd.read_csv(conflict_zones)
    # conflict_zones['Valid until'] #one is not in effect last month of september
    df_cz['Valid until'] = df_cz['Valid until'].fillna('01/01/2055')
    df_cz['Valid until'] = pd.to_datetime(df_cz['Valid until'], format="%d/%m/%Y")
    df_cz['Issue date'] = pd.to_datetime(df_cz['Issue date'], format="%d/%m/%Y")
    df_cz.loc[df_cz['Affected Country'].isna(), 'Affected Country'] = (
        df_cz.loc[df_cz['Affected Country'].isna(), 'Subject'].str.extract('Airspace of (.*)').values)
    df_cz = df_cz[['Valid until', 'Issue date', 'Affected Country']]
    df_cz.columns = ['end', 'start', 'country']
    df_cz.country = coco.convert(df_cz.country, to='ISO2')
    df_cz = df_cz.explode('country')
    df_cz = df_cz[df_cz.start < pd.to_datetime('2023/01/01')]

    df_cz = df_cz.merge(ldf, how='left')
    df_cz = gpd.GeoDataFrame(df_cz)
    # seems there are 3 different ones, 4 ranges
    start = pd.to_datetime('2021/12/31')
    stop = pd.to_datetime('2023/01/01')
    splitters = df_cz.start[(df_cz.start < stop) & (df_cz.start > start)]
    splitters2 = df_cz.end[(df_cz.end < stop) & (df_cz.end > start)]
    splitters = pd.concat([splitters2, splitters, pd.Series(start), pd.Series(stop)]).sort_values()
    ranges = [(splitters.iloc[0+i], splitters.iloc[1+i]) for i in range(len(splitters)-1)]

    # put circle of the following 4 ETOPSs radius around airports, and do the union
    # ETOPS-90 #ETOPS-120(/138) #ETOPS-180(/207)
    airports = airports[airports.type.isin(['medium_airport', 'large_airport'])]
    airports['geometry'] = gpd.points_from_xy(airports.longitude_deg, airports.latitude_deg)
    airports = airports[['iso_country', 'geometry']]

    times = np.array([60, 90, 120, 138, 180, 207])
    # 430kts taken, etops depends onweight and type of plane, but its out of scope for me
    radii = times*400*1.852/60
    etops = []
    excluded_countries = []
    for i, time_etop in enumerate(ranges):
        excluded = df_cz.country[((df_cz.start <= time_etop[0]) & (df_cz.end >= time_etop[1]))]
        excluded_countries += [excluded.values]
        airports_etop = airports[~airports.iso_country.isin(excluded_countries)]
        airports_etop = gpd.GeoDataFrame(airports_etop)

        ae = []
        for r, radius in enumerate(radii):
            print(i, ':', r)
            ae += [draw_buffer_around_points(airports_etop, radius=radius, steps=16)]

        ae = pd.concat(ae)
        ae['etops_rating'] = times
        etops += [ae]

    splitters = [date.tz_localize('UTC') for date in splitters]
    ranges = [(splitters[0+i], splitters[1+i]) for i in range(len(splitters)-1)]
    return etops, ranges
    # runs in <60s
