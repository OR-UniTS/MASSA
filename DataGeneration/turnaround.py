import pandas as pd
import multiprocessing as mp

df_turn_around = pd.read_csv('data/turnaround_2019.csv')


def get_critical(args):
    critical_list = []
    df_day, series_day, airports, week_num = args
    infeasible = 0
    not_critical = 0

    for aircraft in df_day.icao24.unique():
        df_aircraft = df_day[df_day.icao24 == aircraft]
        arrival_previous_airport = df_aircraft.arrival.iloc[0]
        arr_time = df_aircraft.arr_time.iloc[0]
        arr_id = df_aircraft.id_arr.iloc[0]
        wide_body = df_aircraft.wide_body.iloc[0]
        min_turnaround = 90 if wide_body else 30
        for i in range(1, df_aircraft.shape[0]):
            dep_airport = df_aircraft.departure.iloc[i]
            dep_time = df_aircraft.dep_time.iloc[i]
            dep_id = df_aircraft.id_dep.iloc[i]

            if dep_airport == arrival_previous_airport and arrival_previous_airport in airports \
                    and dep_id in series_day and arr_id in series_day:
                if arr_time + 30 > dep_time - 30 - min_turnaround:
                    if arr_time <= dep_time - min_turnaround:  # discard infeasible
                        critical_list.append([dep_id, arr_id, week_num, wide_body])
                    else:
                        infeasible += 1
                else:
                    not_critical += 1
            arrival_previous_airport = dep_airport
            arr_time = df_aircraft.arr_time.iloc[i]
            arr_id = df_aircraft.id_arr.iloc[i]
    # print(infeasible)
    return critical_list


def get_turnaround(db_slot):
    airports = db_slot.airport.unique()
    df = df_turn_around.copy(deep=True)

    df['code'] = df.icao24 + '-' + df.day_num.astype(str)
    df_turnaround = df[df.duplicated(subset='code', keep=False)].copy(deep=True)
    df_turnaround = df_turnaround[~df_turnaround.duplicated()]

    week_num = dict(zip(sorted(df_turnaround.day_num.unique()), range(df_turnaround.day_num.unique().shape[0])))
    df_turnaround['week_num'] = df_turnaround.day_num.apply(lambda d: week_num[d])

    series_time = dict(zip(db_slot.id, db_slot.Time))
    series_time[-1] = -1

    df_turnaround['dep_time'] = df_turnaround.id_dep.apply(lambda i: series_time[i])
    df_turnaround['arr_time'] = df_turnaround.id_arr.apply(lambda i: series_time[i])

    df_turnaround.sort_values(by=['code', 'week_num', 'dep_time_original'], inplace=True)
    pd.set_option('display.max_columns', None)

    args = [(df_turnaround[df_turnaround.week_num == week_num],
             db_slot[(db_slot.InitialDate <= week_num) & (week_num <= db_slot.FinalDate)].id, airports, week_num) for
            week_num in range(30)]

    with mp.Pool() as pool:
        res = pool.map(get_critical, args)
    critical = [el for r in res for el in r]
    df_turn = pd.DataFrame(columns=['departure', 'arrival', 'day', 'wide_body'], data=critical)
    # df_turn.to_csv('turnaround.csv', index_label=False, index=False)
    # db_slot.to_csv('db_slot.csv', index_label=False, index=False)

    # df_turn_new, db_slot_new = fix_turn_time(df_turn, db_slot)
    return df_turn
