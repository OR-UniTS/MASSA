import pandas as pd
import numpy as np
import datetime

import pytz

pd.set_option('display.max_columns', None)


def rename(df_in):
    renamed = df_in[["icao24", "firstseen", "estdepartureairport", "lastseen", "estarrivalairport", "callsign",
                     "estdepartureairporthorizdistance", "estdepartureairportvertdistance",
                     "estarrivalairporthorizdistance",
                     "estarrivalairportvertdistance", "departureairportcandidatescount",
                     "arrivalairportcandidatescount",
                     "day"]].copy()
    renamed.columns = ['icao24', "dep_time", 'departure', "arr_time", 'arrival', 'callsign', 'dep dist', 'dep alt',
                       'arr dist', 'arr alt', 'candidate dep airports', 'candidate arr airports', 'day']
    return renamed[['icao24', "dep_time", 'departure', "arr_time", 'arrival', 'callsign', 'day']]


def filter_airports(df_raw):
    # df = pd.read_csv(df_raw, sep="\t")
    df_airports = pd.read_csv("SeriesDetection/RawData/airports.csv", index_col=None).drop(columns="Unnamed: 0")
    df_airports = df_airports[df_airports.level == 3]
    df_eu = df_raw[df_raw.departure.isin(df_airports.airport)
                   | df_raw.arrival.isin(df_airports.airport)]
    df_eu = df_eu[df_eu.departure != df_eu.arrival].copy(deep=True)
    df_eu.departure = df_eu.departure.apply(lambda dep: "UNKNOWN" if type(dep) == float else dep)
    df_eu.arrival = df_eu.arrival.apply(lambda arr: "UNKNOWN" if type(arr) == float else arr)

    return df_eu


def day_converter(df_eu):
    df_eu.sort_values(by="day", inplace=True, ignore_index=True)
    df_eu["week_day"] = df_eu["day"].apply(lambda d: datetime.datetime.fromtimestamp(d).weekday())
    df_eu["day"] = df_eu["day"].apply(lambda d: str(datetime.datetime.fromtimestamp(d))[:10])
    return df_eu


def time_minute_conversion(df_eu):
    time_ = df_eu["dep_time"].apply(
        lambda d: datetime.datetime.fromtimestamp(d, tz=pytz.UTC).time() if not np.isnan(d) else "NaN")
    df_eu["dep_time"] = time_
    time_ = df_eu["arr_time"].apply(
        lambda d: datetime.datetime.fromtimestamp(d, tz=pytz.UTC).time() if not np.isnan(d) else "NaN")
    df_eu["arr_time"] = time_
    t_min = df_eu["dep_time"].apply(
        lambda t: np.round(t.hour * 60 + t.minute + t.second * 0.1) if type(t) == datetime.time else 0).astype(int)
    df_eu["dep_min"] = t_min
    t_min = df_eu["arr_time"].apply(
        lambda t: np.round(t.hour * 60 + t.minute + t.second * 0.1) if type(t) == datetime.time else 0).astype(int)
    df_eu["arr_min"] = t_min
    return df_eu


def from_raw_to_season(df_raw_name, year, save=False):
    df_raw = pd.read_csv(df_raw_name)
    df_raw = rename(df_raw)

    final_df = filter_airports(df_raw)
    final_df = day_converter(final_df)

    start = datetime.datetime(year, 3, 31) if year == 2019 else datetime.datetime(year, 4, 1)
    end = datetime.datetime(year, 10, 27) if year == 2019 else datetime.datetime(year, 10, 28)

    final_df = final_df[(pd.to_datetime(final_df["day"]) >= start) &
                        (pd.to_datetime(final_df["day"]) < end)]
    final_df = time_minute_conversion(final_df)
    final_df["airline"] = final_df["callsign"].apply(lambda call: call[:3])

    d1 = datetime.date(2019, 3, 31) if year == 2019 else datetime.date(2018, 4, 1)
    d2 = datetime.date(2019, 10, 27) if year == 2019 else datetime.date(2018, 10, 28)
    dd = sorted([(d1 + datetime.timedelta(days=x)).strftime("%Y-%m-%d") for x in range((d2 - d1).days + 1)])
    days = dict(zip(dd, range(len(dd))))
    final_df['day_num'] = final_df.day.apply(lambda d: days[d])

    final_df.callsign = final_df.callsign.apply(lambda call: call.strip())
    final_df = final_df[final_df.callsign.str.contains('^[A-Z]{3}[0-9]{1,4}[A-Z]{0,2}$', regex=True)].copy(deep=True)

    final_df['od'] = final_df.airline + '-' + final_df.departure + '-' + final_df.arrival
    final_df['od_day'] = final_df.od + '-' + final_df.day_num.astype(str)
    final_df['od_week_day'] = final_df.od + '-' + final_df['week_day'].astype(str)
    final_df['od_call_week_day'] = final_df.od + '-' + final_df['week_day'].astype(str) + '-' + final_df['callsign']

    if save:
        final_df.to_csv("SeriesDetection/DataSummer/summer_" + str(year) + ".csv", index_label=False, index=False)
    else:
        # print(final_df)
        return final_df


from_raw_to_season("SeriesDetection/RawData/flights_2018.csv", 2018, save=True)
from_raw_to_season("SeriesDetection/RawData/flights_2019.csv", 2019, save=True)
