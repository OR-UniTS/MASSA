import numpy as np
import pandas as pd

from DictAndRanges.dicts_and_ranges import AirportParameters

WEEK_DAY = 4
DAY_NUM = WEEK_DAY + 1

df = pd.read_csv('data/friday_2019.csv')
airports = df.A_ICAO.unique()

df_airports = pd.read_csv('data/airports.csv')
# df['matched_slot'] = -1
air_p = dict(zip(df_airports.airport, [AirportParameters(slot_size=slot_size) for slot_size in df_airports['slot size']]))

h_caps = dict(zip(airports, [dict(zip(['G', 'D', 'A'], [np.zeros((24, 30), dtype=int) for _ in range(3)])) for _ in airports]))
hi_caps = dict(zip(airports, [dict(zip(['G', 'D', 'A'], [np.zeros((288, 30), dtype=int) for _ in range(3)])) for _ in airports]))
i_caps = dict(zip(airports, [np.zeros((288, 30), dtype=int) for _ in airports]))
series_airport_dict = {}

x_idx, y_idx = 0, 0
for i, airport in enumerate(airports):
    print(airport)
    df_airport = df[df.A_ICAO == airport]
    series_airport_dict[airport] = []
    p = air_p[airport]
    for _, ser in df_airport.iterrows():
        h_caps[airport]['G'][p.unit_to_hours[ser.Slot], ser.InitialDate: ser.FinalDate + 1] += 1
        hi_caps[airport]['G'][p.unit_to_hour_intervals[ser.Slot], ser.InitialDate: ser.FinalDate + 1] += 1
        i_caps[airport][p.unit_to_intervals[ser.Slot], ser.InitialDate: ser.FinalDate + 1] += 1
        if ser.Flow == 'D':
            h_caps[airport]['D'][p.unit_to_hours[ser.Slot], ser.InitialDate: ser.FinalDate + 1] += 1
            hi_caps[airport]['D'][p.unit_to_hour_intervals[ser.Slot], ser.InitialDate: ser.FinalDate + 1] += 1
        else:
            h_caps[airport]['A'][p.unit_to_hours[ser.Slot], ser.InitialDate: ser.FinalDate + 1] += 1
            hi_caps[airport]['A'][p.unit_to_hour_intervals[ser.Slot], ser.InitialDate: ser.FinalDate + 1] += 1

        x_idx += 1

# df_volato = pd.read_csv('data/cap_volato.csv')
# df_volato_max = df_volato.groupby(['airport', 'start', 'flow'], as_index=False).agg('max')[['airport', 'start', 'flow', 'capH']]
# df_volato_max.rename(columns={'capH': 'from_flown'}, inplace=True)
# df_volato_max['from_series'] = df_volato_max.apply(
#     lambda row: h_caps[row.airport][row.flow][row.start // 60].max() if row.airport in airports else 0, axis=1)
# df_volato_max['capH'] = df_volato_max.apply(lambda row: max(row.from_flown, row.from_series), axis=1)
#
# df_volato_max[['from_flown', 'from_series']].max(axis=1)
# df_volato_max.to_csv('data/hourly_sequential_capacity.csv', index_label=False, index=False)

df_volato = pd.read_csv('SeriesDetection/DataSummer/summer_2019.csv')
df_volato = df_volato[df_volato.week_day == WEEK_DAY].copy(deep=True)
df_volato['unit_arr'] = df_volato.arr_min // 5
df_volato['unit_dep'] = df_volato.dep_min // 5
df_volato['day_idx'] = (df_volato.day_num - DAY_NUM) // 7

h_caps_v = dict(zip(airports, [dict(zip(['G', 'D', 'A'], [np.zeros((24, 30), dtype=int) for _ in range(3)])) for _ in airports]))
hi_caps_v = dict(zip(airports, [dict(zip(['G', 'D', 'A'], [np.zeros((288, 30), dtype=int) for _ in range(3)])) for _ in airports]))
i_caps_v = dict(zip(airports, [np.zeros((288, 30), dtype=int) for _ in airports]))

air_p: dict[str, AirportParameters]
for airport in airports:
    print(airport)
    df_dep = df_volato[df_volato.departure == airport]
    for day in df_dep.day_idx.unique():
        for hour in range(24):
            h_caps_v[airport]['G'][hour, day] += df_dep[df_dep.unit_dep.isin(air_p[airport].unit_to_hours[hour])].shape[0]
            h_caps_v[airport]['D'][hour, day] += df_dep[df_dep.unit_dep.isin(air_p[airport].unit_to_hours[hour])].shape[0]
        for unit in range(288):
            hi_caps_v[airport]['G'][unit, day] += df_dep[df_dep.unit_dep.isin(air_p[airport].unit_to_hour_intervals[unit])].shape[0]
            hi_caps_v[airport]['D'][unit, day] += df_dep[df_dep.unit_dep.isin(air_p[airport].unit_to_hour_intervals[unit])].shape[0]
            i_caps_v[airport][unit, day] += df_dep[df_dep.unit_dep.isin(air_p[airport].unit_to_intervals[unit])].shape[0]

    df_arr = df_volato[df_volato.arrival == airport]
    for day in df_arr.day_idx.unique():
        for hour in range(24):
            h_caps_v[airport]['G'][hour, day] += df_arr[df_arr.unit_dep.isin(air_p[airport].unit_to_hours[hour])].shape[0]
            h_caps_v[airport]['A'][hour, day] += df_arr[df_arr.unit_dep.isin(air_p[airport].unit_to_hours[hour])].shape[0]
        for unit in range(288):
            hi_caps_v[airport]['G'][unit, day] += df_arr[df_arr.unit_dep.isin(air_p[airport].unit_to_hour_intervals[unit])].shape[0]
            hi_caps_v[airport]['A'][unit, day] += df_arr[df_arr.unit_dep.isin(air_p[airport].unit_to_hour_intervals[unit])].shape[0]
            i_caps_v[airport][unit, day] += df_arr[df_arr.unit_dep.isin(air_p[airport].unit_to_intervals[unit])].shape[0]

df_caps_rolling = pd.DataFrame({'airport': airports, 'rolling_cap': [max(i_caps[a].max(), i_caps_v[a].max()) for a in airports],
                                'rolling_cap_hour_G': [max(hi_caps[a]['G'].max(), hi_caps_v[a]['G'].max()) for a in airports],
                                'rolling_cap_hour_D': [max(hi_caps[a]['D'].max(), hi_caps_v[a]['D'].max()) for a in airports],
                                'rolling_cap_hour_A': [max(hi_caps[a]['A'].max(), hi_caps_v[a]['A'].max()) for a in airports]})
df_caps_rolling.to_csv('data/rolling_cap.csv', index_label=False, index=False)

df_hourly_sequential = pd.DataFrame(columns=['airport', 'hour', 'flow', 'capH'])
for airport in airports:
    for flow in ['G', 'D', 'A']:
        df_hourly_sequential = pd.concat([df_hourly_sequential,
                                          pd.DataFrame({'airport': [airport for _ in range(24)], 'hour': range(24),
                                                        'flow': [flow for _ in range(24)],
                                                        'capH': np.maximum(h_caps[airport][flow].max(axis=1),
                                                                           h_caps[airport][flow].max(axis=1))})],
                                         ignore_index=True)

df_hourly_sequential.to_csv('data/hourly_sequential_capacity.csv', index_label=False, index=False)
