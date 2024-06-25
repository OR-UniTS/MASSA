import numpy as np
import pandas as pd

from DictAndRanges.dicts_and_ranges import Parameters
from SeriesDetection.df_utils import fix_df


def make_turnaround_df(df_init, df_series):
    df_turnaround = df_init[df_init.od_call_week_day.isin(df_series.od_call_week_day)].copy(deep=True)

    week_num = dict(zip(sorted(df_init.day_num), range(df_init.day_num.shape[0])))
    df_init['week_num'] = df_init.day_num.apply(lambda d: week_num[d])

    df_aircraft = pd.read_csv('SeriesDetection/RawData/aircraftDatabase.csv')
    wide_body_list = ['B763', 'B744', 'A332']
    type_dict = dict(zip(df_aircraft.icao24, df_aircraft.typecode))
    df_turnaround['type'] = df_turnaround.icao24.apply(
        lambda icao: type_dict[icao] if icao in type_dict.keys() else 'none')

    df_cluster = pd.read_csv('SeriesDetection/RawData/aircraftClustering.csv')
    cluster = dict(zip(df_cluster.AircraftType, df_cluster.AssignedAircraftType))
    df_turnaround.type = df_turnaround.type.apply(lambda t: cluster[t] if t in cluster.keys() else 'none')
    df_turnaround['wide_body'] = df_turnaround.type.apply(lambda t: True if t in wide_body_list else False)

    dep_series_id = dict(zip(df_series[df_series.Flow == 'D'].od_call_week_day, df_series[df_series.Flow == 'D'].id))
    arr_series_id = dict(zip(df_series[df_series.Flow == 'A'].od_call_week_day, df_series[df_series.Flow == 'A'].id))
    df_turnaround['id_dep'] = df_turnaround.od_call_week_day.apply(
        lambda o: dep_series_id[o] if o in dep_series_id.keys() else -1)
    df_turnaround['id_arr'] = df_turnaround.od_call_week_day.apply(
        lambda o: arr_series_id[o] if o in arr_series_id.keys() else -1)
    df_turnaround = df_turnaround[
        ['icao24', 'dep_time', 'day_num', 'id_dep', 'id_arr', 'wide_body', 'departure', 'arrival']]
    df_turnaround.rename(columns={'dep_time': 'dep_time_original'}, inplace=True)
    df_turnaround.to_csv('data/turnaround_2019.csv', index_label=False, index=False)


def make_db_slot(df_init, turnaround=False):
    df_airports = pd.read_csv("SeriesDetection/RawData/airports.csv", index_col=None).drop(columns="Unnamed: 0")
    df_airports = df_airports[df_airports.level == 3]

    df_init['odaw'] = df_init.od + '-' + df_init.airline + df_init['week_day'].astype(str)
    # df_init['odawd'] = df_init.od + '-' + df_init.airline + df_init['week_day'].astype(str) + df_init['day_num'].astype(str)

    df_series_not_filtered = df_init.groupby(['od_call_week_day'], as_index=False).agg(
        {'od_call_week_day': ['min', 'count'], 'dep_min': ['mean', 'median', 'std'],
         'arr_min': ['mean', 'median', 'std'],
         'day_num': ['min', 'max'], 'departure': 'min', 'arrival': 'min', 'week_day': 'min', 'airline': 'min'})

    df_series_not_filtered.columns = ['od_call_week_day', 'n_repetitions', 'dep_mean', 'dep_median', 'dep_std',
                                      'arr_mean', 'arr_median', 'arr_std', 'start', 'end', 'departure', 'arrival',
                                      'week_day', 'Airline']

    df_series_not_filtered['n_slots'] = ((df_series_not_filtered.end - df_series_not_filtered.start) / 7).astype(
        int) + 1

    df_series = df_series_not_filtered[df_series_not_filtered.n_slots >= 5]
    df_series = df_series[df_series.n_repetitions >= 3].copy(deep=True)

    df_series['dep_min'] = df_series.dep_median.apply(lambda t: int(5 * np.round(t / 5)))
    df_series['arr_min'] = df_series.arr_median.apply(lambda t: int(5 * np.round(t / 5)))
    df_series['matched'] = -1
    df_series['Flow'] = 'N'
    df_series['id'] = range(df_series.shape[0])
    df_series['InitialDate'] = df_series.start // 7
    df_series['FinalDate'] = df_series.end // 7
    df_series['airport'] = 'N'
    df_series['other_airport'] = 'N'
    df_series['Time'] = -1

    dep_condition = df_series.departure.isin(df_airports.airport)
    arr_condition = df_series.arrival.isin(df_airports.airport)

    df_series.loc[dep_condition, 'Flow'] = 'D'
    df_series.loc[~dep_condition & arr_condition, 'Flow'] = 'A'

    df_series.loc[dep_condition, 'Time'] = df_series[dep_condition].dep_min
    df_series.loc[~dep_condition & arr_condition, 'Time'] = df_series[~dep_condition & arr_condition].arr_min

    df_series.loc[dep_condition, 'airport'] = df_series[dep_condition].departure
    df_series.loc[dep_condition, 'other_airport'] = df_series[dep_condition].arrival
    df_series.loc[~dep_condition & arr_condition, 'airport'] = df_series[~dep_condition & arr_condition].arrival
    df_series.loc[~dep_condition & arr_condition, 'other_airport'] = df_series[~dep_condition & arr_condition].departure

    # exceptions arrival < departure: here in the raw df the day is the arrival day

    arr_before_dep_condition = df_series.arr_min <= df_series.dep_min
    exception_condition = dep_condition & arr_condition & arr_before_dep_condition
    df_series.loc[exception_condition, 'Flow'] = 'A'
    df_series.loc[exception_condition, 'Time'] = df_series[exception_condition].arr_min
    df_series.loc[exception_condition, 'airport'] = df_series[exception_condition].arrival
    df_series.loc[exception_condition, 'other_airport'] = df_series[exception_condition].departure

    # matching

    match_condition = dep_condition & arr_condition & ~arr_before_dep_condition
    df_matching = df_series[match_condition].copy(deep=True)
    df_matching.matched = df_series[match_condition].id
    df_matching.Flow = 'A'
    df_matching.Time = df_matching.arr_min
    df_matching.airport = df_matching.arrival
    df_matching.id = range(df_series.shape[0], df_series.shape[0] + df_matching.shape[0])

    df_series = pd.concat([df_series, df_matching], ignore_index=True)

    if turnaround:
        make_turnaround_df(df_init, df_series)

    return df_series


def get_historic_raw(raw, ser_2018):
    # print(raw.id)
    df_his_18 = ser_2018[(ser_2018.od_air_week_day == raw.od_air_week_day) &
                         (ser_2018.Flow == raw.Flow) &
                         ((raw.Time - 60 <= ser_2018.Time) & (ser_2018.Time <= raw.Time + 60)) &
                         ~ser_2018.Historic]
    if df_his_18.shape[0] > 0:
        ser_2018.at[df_his_18.index[0], 'Historic'] = True
        return True
    else:
        return False


def get_historic(s_2019, s_2018):
    s_2018['Historic'] = False
    s_2019['Historic'] = False
    s_2018[
        'od_air_week_day'] = s_2018.departure + s_2018.arrival + s_2018.Airline + s_2018.week_day.astype(
        str)
    s_2019[
        'od_air_week_day'] = s_2019.departure + s_2019.arrival + s_2019.Airline + s_2019.week_day.astype(
        str)

    s_2019['Historic'] = s_2019.apply(get_historic_raw, axis=1, ser_2018=s_2018)
    s_2019 = fix_df(s_2019)
    parameters = Parameters()
    s_2019['Slot'] = s_2019.Time.apply(lambda t: parameters.time_to_unit[t])
    s_2019['Hour'] = s_2019.Time.apply(lambda t: parameters.time_to_hour[t])
    return s_2019[['id', 'Airline', 'airport', 'other_airport', 'Time', 'InitialDate', 'FinalDate', 'matched', 'Flow', 'week_day',
                   'Historic', 'HistoricChanged', 'HistoricOriginalTime', 'HistoricOriginalSlot', 'Slot', 'Hour']]


df_init_18 = pd.read_csv('SeriesDetection/DataSummer/summer_2018.csv')
df_init_19 = pd.read_csv('SeriesDetection/DataSummer/summer_2019.csv')

friday_18 = df_init_18[df_init_18.week_day == 4].copy(deep=True)
friday_18 = make_db_slot(friday_18, turnaround=False)

friday_18.to_csv('data/friday_2018.csv', index_label=False, index=False)

friday_19 = df_init_19[df_init_19.week_day == 4].copy(deep=True)
friday_19 = make_db_slot(friday_19, turnaround=True)

friday_19 = get_historic(friday_19, friday_18)

friday_19.to_csv('data/friday_2019.csv', index_label=False, index=False)

# for i in range(1, 7):
#     friday_18_ = df_init_18[df_init_18.week_day == i].copy(deep=True)
#     friday_18_ = make_db_slot(friday_18_, turnaround=False)
#
#     friday_19_ = df_init_19[df_init_19.week_day == i].copy(deep=True)
#     friday_19_ = make_db_slot(friday_19_, turnaround=True)
#
#     friday_19_ = get_historic(friday_19_, friday_18_)
#     friday_19 = pd.concat([friday_19, friday_19_], ignore_index=True)
#
# friday_19.to_csv('data/series_all_season.csv', index_label=False, index=False)


