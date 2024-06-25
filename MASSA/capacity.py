import numpy as np
import pandas as pd
import multiprocessing as mp
from DictAndRanges.dicts_and_ranges import Parameters, AirportParameters


def get_hourly_parallel(df_airp, df_capacity, p, correction):
    capacity = np.repeat(
        np.expand_dims(df_capacity.sort_values(by='hour').capH, axis=1), 30, axis=1)

    for day in p.days:
        non_allocable_day = df_airp[(df_airp.InitialDate <= day) & (day <= df_airp.FinalDate)]
        if non_allocable_day.shape[0] > 0:
            capacity_used = non_allocable_day.groupby(['Slot'], as_index=False).agg({'Slot': ['min', 'count']})
            for hour in range(24):
                capacity[hour, day] -= (
                    capacity_used[capacity_used[('Slot', 'min')].isin(p.hour_to_units[hour])][('Slot', 'count')].sum())
            # capacity[capacity_used[('Slot', 'min')].to_list(), day] += -capacity_used['interval']

    if correction > 0:
        capacity = np.abs(capacity) + correction
    return capacity


def get_actual_capacity(flow, airports, series_historic, df_capacity, params: dict[str, AirportParameters], correction):
    series_flow = series_historic[series_historic.Flow == flow] if flow != 'G' else series_historic

    args = [(series_flow[series_flow.airport == airport],
             df_capacity[(df_capacity.airport == airport) & (df_capacity.flow == flow)], params[airport], correction)
            for airport in airports]

    with mp.Pool() as pool:
        res = pool.starmap(get_hourly_parallel, args)

    capacity = dict(zip(airports, res))

    return capacity


def get_interval_parallel(df_airp, interval_cap, p, correction, hour_interval):
    interval = p.unit_to_hour_intervals if hour_interval else p.unit_to_intervals
    capacity = np.ones((288, 30), dtype=int) * interval_cap
    for day in p.days:
        non_allocable_day = df_airp[(df_airp.InitialDate <= day) & (day <= df_airp.FinalDate)]
        if non_allocable_day.shape[0] > 0:
            capacity_used = non_allocable_day.groupby(['Slot'], as_index=False).agg({'Slot': ['min', 'count']})
            for unit in range(288):
                capacity[unit, day] -= (
                    capacity_used[capacity_used[('Slot', 'min')].isin(interval[unit])][('Slot', 'count')].sum())
            # capacity_used['interval'] = \
            #     capacity_used[('Slot', 'min')].apply(
            #         lambda s: capacity_used[capacity_used[('Slot', 'min')].isin(interval[s])]
            #         [('Slot', 'count')].sum())
            # capacity[capacity_used[('Slot', 'min')].to_list(), day] += -capacity_used['interval']

    if correction > 0:
        capacity = np.abs(capacity) + correction

    return capacity


def get_interval_capacity_parallel(flow, hour_interval, airports, series_historic, df_intervals, p: dict[str, AirportParameters],
                                   correction):
    series_flow = series_historic[series_historic.Flow == flow] if flow != 'G' else series_historic
    args = [(series_flow[series_flow.airport == airport], df_intervals[df_intervals.airport == airport]
    ['rolling_cap' if not hour_interval else ('rolling_cap_hour_G' if flow == 'G' else
                                              ('rolling_cap_hour_D' if flow == 'D' else 'rolling_cap_hour_A'))].iloc[0],
             p[airport], correction, hour_interval) for airport in airports]

    with mp.Pool() as pool:
        res = pool.starmap(get_interval_parallel, args)

    capacity = dict(zip(airports, res))
    return capacity


def get_hour_capacities(series, series_historic, parameters: dict[str, AirportParameters], correction=0):
    if correction > 0:
        print('applied capacity correction of +', correction)
    capacities = []
    airports = series.airport.unique()

    df_capacity = pd.read_csv('data/hourly_sequential_capacity.csv')
    # df_capacity['Slot'] = df_capacity.start.apply(lambda t: parameters.time_to_unit[t])
    # df_capacity['Hour'] = df_capacity.start//60

    for flow in ['G', 'D', 'A']:
        capacities.append(get_actual_capacity(flow, airports, series_historic, df_capacity, parameters, correction))
    return capacities


def get_hour_interval_capacities(series, series_historic, parameters: dict[str, AirportParameters], correction=0):
    capacities = []
    airports = series.airport.unique()
    hour_interval = True
    df_intervals = pd.read_csv('data/rolling_cap.csv')
    for flow in ['G', 'D', 'A']:
        capacities.append(get_interval_capacity_parallel(
            flow, hour_interval, airports, series_historic, df_intervals, parameters, correction))
    return capacities


def get_interval_capacities(series, series_historic, parameters: dict[str, AirportParameters], correction=0):
    airports = series.airport.unique()
    hour_interval = False
    df_intervals = pd.read_csv('data/rolling_cap.csv')
    flow = 'G'
    return get_interval_capacity_parallel(flow, hour_interval, airports, series_historic, df_intervals, parameters, correction)
