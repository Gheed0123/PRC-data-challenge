# -*- coding: utf-8 -*-
"""
Created by Herpa Derp

only vectorizing ones I use ofcourse
and removed error checking
"""
import numpy as np
import pandas as pd
from collections import namedtuple
from numpy import sin, cos, pi, sqrt
from numpy import arctan2 as atan2
from numpy import arccos as acos
from shapely.geometry import Polygon, LineString

radius_earth_meters = 6371000

eligible_units = ['meters', 'kilometers', 'miles', 'feet', 'yards', 'nautical_miles']
Units = namedtuple('Units', field_names=eligible_units)
radius_earth = Units(
    meters=radius_earth_meters,
    kilometers=radius_earth_meters / 1000,
    miles=radius_earth_meters / 1609.344,
    feet=radius_earth_meters * 3.28084,
    nautical_miles=radius_earth_meters / 1852,
    yards=radius_earth_meters * 1.09361
)


def intermediate_point(p1, p2, fraction=0.5, delta=pd.Series()):
    """ This function calculates the intermediate point along the course laid out by p1 to p2.  fraction is the fraction
    of the distance between p1 and p2, where 0 is p1, 0.5 is equivalent to midpoint(*), and 1 is p2.
    :param p1: tuple point of (lon, lat)
    :param p2: tuple point of (lon, lat)
    :param fraction: the fraction of the distance along the path.
    :return: point (lon, lat)
    """
    p11 = np.radians(p1)
    p22 = np.radians(p2)
    lon1, lat1 = p11.longitude.values, p11.latitude.values
    lon2, lat2 = p22.longitude.values, p22.latitude.values
    if (len(delta) == 0):
        delta = distance_between_points(p1, p2)
    delta = delta/radius_earth.meters*2*pi
    a = sin((1 - fraction) * delta) / sin(delta)
    b = sin(fraction * delta) / sin(delta)
    x = a * cos(lat1) * cos(lon1) + b * cos(lat2) * cos(lon2)
    y = a * cos(lat1) * sin(lon1) + b * cos(lat2) * sin(lon2)
    z = a * sin(lat1) + b * sin(lat2)
    lat3 = atan2(z, sqrt(x * x + y * y)).reset_index(drop=True)
    lon3 = atan2(y, x).reset_index(drop=True)
    return np.degrees([lon3, lat3])


def distance_between_points(p1, p2, unit='meters', func='haversine'):
    """ This function computes the distance between two points in the unit given in the unit parameter.  It will
    calculate the distance using the haversine unless the user specifies haversine to be False.  Then law of cosines
    will be used
    :param p1: tuple point of (lon, lat)
    :param p2: tuple point of (lon, lat)
    :param unit: unit of measurement. List can be found in constants.eligible_units
    :param haversine: True (default) uses haversine distance, False uses law of cosines
    :return: Distance between p1 and p2 in the units specified.
    https://github.com/seangrogan/great_circle_calculator
    """
    p11 = np.radians(p1)
    p22 = np.radians(p2)
    lon1, lat1 = p11.longitude, p11.latitude
    lon2, lat2 = p22.longitude, p22.latitude
    r_earth = getattr(radius_earth, unit, 'meters')

    if (func == 'haversine'):
        # Haversine
        d_lat, d_lon = lat2 - lat1, lon2 - lon1
        a = sin(d_lat / 2) * sin(d_lat / 2) + cos(lat1) * cos(lat2) * sin(d_lon / 2) * sin(d_lon / 2)
        a = np.minimum(1, a)
        c = 2 * atan2(sqrt(a), sqrt((1 - a)))
        dist = r_earth * c
        return dist

    # Spherical Law Of Cosines
    dist = acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2 - lon1)) * r_earth
    return dist


def draw_buffer_around_points(gdf, radius=50, steps=np.pi*10):
    # radius in km!!
    # crude geodesic buffer
    lon = np.radians(gdf.geometry.x.values)
    lat = np.radians(gdf.geometry.y.values)

    circumference = radius/(radius_earth.kilometers*2*pi)*2*pi
    stepSize = (1/round(steps))*2*pi
    latstep = circumference

    lonsteps = (circumference / cos(lat))
    # lonsteps[lonsteps>(2*pi)]=2*pi
    # lonsteps[lonsteps<(-2*pi)]=2*pi
    step = np.array(range(round(steps)))*stepSize

    lats = np.degrees(((cos(step)*latstep)[:, np.newaxis])+lat).T
    lons = np.degrees((sin(step)[:, np.newaxis]*(lonsteps))+lon).T

    lons[lons > 180] = 180
    lons[lons < -180] = -180
    polygons = [Polygon(zip(x, lats[i])) for i, x in enumerate(lons)]
    polygdf = gdf.copy()
    polygdf.geometry = polygons
    # polygdf[:100000].plot()
    polygdf = polygdf.dissolve()
    return polygdf


def check_etops(flights, day, etops, ranges):
    i = [i for i, rang in enumerate(ranges) if (rang[0] <= day) & (rang[1] > day)][0]
    etop = etops[i]
    maxetops = len(etops[0])

    # slow, apparently shapely old v 2.0.1 was the cause
    etops_rating = [LineString(zip(flight.longitude, flight.latitude)).within(etop.geometry).sum() if (len(flight) > 2) else maxetops for flight in flights]
    return etops_rating
