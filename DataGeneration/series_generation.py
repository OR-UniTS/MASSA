import numpy as np
import pandas as pd


def matched_shift(row):
    if row.HistoricChanged:
        max_back_shift = max([- row.Time, - row.match_time, -100])
        max_for_shift = min([1435 - row.Time, 1435 - row.match_time, 100])
        times = [t for t in range(max_back_shift, max(max_back_shift, -30), 5)] + \
                [t for t in range(min(max_for_shift, 40), max_for_shift + 1, 5)]
        return np.random.choice(times)
    else:
        return 0


def new_time(row):
    if row.HistoricChanged:
        max_back_shift = max([- row.Time, -100])
        max_for_shift = min([1435 - row.Time, 100])
        times = [t for t in range(-max_back_shift, max(-max_back_shift, -41), 5)] + \
                [t for t in range(min(max_for_shift, 40), max_for_shift + 1, 5)]
        return row.Time + np.random.choice(times)
    else:
        return row.Time


def generate_new_series(matched_series, series_non_historic, airlines, start_idx_new_series, percentage, df_turn):
    n = series_non_historic.shape[0]
    new_matched_series = matched_series.sample(n=int((matched_series.shape[0] * percentage) // 2), replace=False).copy(
        deep=True)
    new_match = series_non_historic[series_non_historic.id.isin(new_matched_series.matched)].copy(deep=True)
    new_matched_series = pd.concat([new_match, new_matched_series], ignore_index=True)
    new_non_matched = series_non_historic[
        (series_non_historic.matched == -1) & (~series_non_historic.id.isin(new_matched_series.id))]
    n_non_matched = int((n - new_matched_series.shape[0]) * percentage)
    new_non_matched = new_non_matched.sample(n=n_non_matched, replace=False).copy(deep=True)

    new_series = pd.concat([new_matched_series, new_non_matched], ignore_index=True)
    new_series_id = dict(
        zip(new_series.id.copy(deep=True), range(start_idx_new_series, start_idx_new_series + new_series.shape[0])))
    new_series_id[-1] = -1
    new_series.id = new_series.id.apply(lambda i: new_series_id[i])
    new_series.matched = new_series.matched.apply(lambda i: new_series_id[i])
    air_dict = dict(zip(new_series.Airline.unique(), np.random.choice(airlines, new_series.Airline.unique().shape[0])))
    new_series.Airline = new_series.Airline.apply(lambda a: air_dict[a])

    df_new_turn = df_turn[
        df_turn.departure.isin(new_series_id.keys()) & df_turn.arrival.isin(new_series_id.keys())].copy(deep=True)

    df_new_turn.departure = df_new_turn.departure.apply(lambda i: new_series_id[i])
    df_new_turn.arrival = df_new_turn.arrival.apply(lambda i: new_series_id[i])
    df_turn = pd.concat([df_turn, df_new_turn], ignore_index=True)

    return new_series, df_turn


def generate_historic_change(sh, percentage_hrc):
    series_historic = sh.copy(deep=True)
    m_series = series_historic[series_historic.matched >= 0]
    idx = m_series.sample(n=int(m_series.shape[0] * 0.5 * percentage_hrc), replace=False).id
    selected_matched_series = m_series[m_series.id.isin(idx)]
    match_idx = selected_matched_series.matched
    match = series_historic[series_historic.id.isin(selected_matched_series.matched)]

    match_time = dict(zip(match.id, match.Time))
    series_historic['match_time'] = series_historic.matched.apply(
        lambda i: match_time[i] if i in match_time.keys() else -1)

    series_historic.loc[series_historic.id.isin(idx), 'HistoricChanged'] = True
    series_historic.loc[series_historic.id.isin(match_idx), 'HistoricChanged'] = True
    series_historic.loc[series_historic.id.isin(idx), 'Historic'] = False
    series_historic.loc[series_historic.id.isin(match_idx), 'Historic'] = False
    series_historic.loc[series_historic.HistoricChanged, 'HistoricOriginalTime'] = \
        series_historic[series_historic.HistoricChanged].Time
    series_historic.loc[series_historic.HistoricChanged, 'HistoricOriginalSlot'] = \
        series_historic[series_historic.HistoricChanged].Slot
    shift = series_historic[series_historic.id.isin(idx)].apply(matched_shift, axis=1)
    series_historic.loc[series_historic.id.isin(idx), 'Time'] += shift
    series_historic.loc[series_historic.id.isin(idx), 'Slot'] = series_historic.loc[
                                                                    series_historic.id.isin(idx)].Time // 5

    m_shift = dict(zip(selected_matched_series.matched, shift))
    series_historic.loc[series_historic.id.isin(match_idx), 'Time'] = \
        series_historic.loc[series_historic.id.isin(match_idx)].apply(lambda row: row.Time + m_shift[row.id],
                                                                      axis=1)
    series_historic.loc[series_historic.id.isin(match_idx), 'Slot'] = \
        series_historic.loc[series_historic.id.isin(match_idx)].Time // 5

    idx = np.concatenate([m_series.matched, m_series.id])

    non_matched = series_historic[
        (series_historic.matched == -1) & (~series_historic.id.isin(idx))]

    idx = non_matched.sample(n=int(non_matched.shape[0] * percentage_hrc), replace=False).id
    series_historic.loc[series_historic.id.isin(idx), 'HistoricChanged'] = True
    series_historic.loc[series_historic.id.isin(idx), 'Historic'] = False
    series_historic.loc[series_historic.id.isin(idx), 'HistoricOriginalTime'] = \
        series_historic[series_historic.id.isin(idx)].Time
    series_historic.loc[series_historic.id.isin(idx), 'HistoricOriginalSlot'] = \
        series_historic[series_historic.id.isin(idx)].Slot
    series_historic.loc[series_historic.id.isin(idx), 'Time'] = \
        series_historic[series_historic.id.isin(idx)].apply(new_time, axis=1)
    series_historic.loc[series_historic.id.isin(idx), 'Slot'] = series_historic[
                                                                    series_historic.id.isin(idx)].Time // 5
    return series_historic


def delete_invalid_turnaround(df_new_turn, s_final):
    time_dict = dict(zip(s_final.id, s_final.Time))
    df_new_turn['dep_time'] = df_new_turn.departure.apply(lambda i: time_dict[i])
    df_new_turn['arr_time'] = df_new_turn.arrival.apply(lambda i: time_dict[i])
    df_new_turn['min_turn'] = 90 * df_new_turn.wide_body + 30 * (1 - df_new_turn.wide_body)
    df_new_turn['turnaround'] = df_new_turn.dep_time - df_new_turn.arr_time - df_new_turn.min_turn

    df_new_turn = df_new_turn[df_new_turn.turnaround >= 0]

    return df_new_turn[['departure', 'arrival', 'day', 'wide_body']].copy()


def get_match_slot(series):
    arrival = series[series.matched != -1]
    departure = series[series.id.isin(arrival.matched)]
    match_time = dict(zip(departure.id, departure.Slot))
    matched_time = dict(zip(arrival.id, arrival.Slot))
    match_idx = dict(zip(arrival.matched, arrival.id))
    matched_idx = dict(zip(arrival.id, arrival.matched))
    series['matched_slot'] = (
        series.id.apply(
            lambda i: matched_time[match_idx[i]] if i in match_idx.keys()
            else (match_time[matched_idx[i]] if i in matched_idx.keys() else -1)))
    return series


def generate_series(series_historic, series_non_historic, airlines, start_idx_new_series, df_turn, percentage,
                    percentage_hrc):
    series_non_historic = series_non_historic.copy()
    series_historic = series_historic.copy()
    matched_series = series_non_historic[series_non_historic.matched >= 0]

    new_series, df_turn = generate_new_series(matched_series, series_non_historic, airlines, start_idx_new_series,
                                              percentage, df_turn)

    s_hrc = generate_historic_change(series_historic, percentage_hrc)
    s_h, s_hrc = (s_hrc[s_hrc.Historic], s_hrc[s_hrc.HistoricChanged])
    s_final = pd.concat([s_hrc, series_non_historic, new_series], ignore_index=True, copy=True)
    s_final.sort_values(by='id', inplace=True)

    s_final = get_match_slot(s_final)
    df_turn = df_turn[df_turn.departure.isin(s_final.id) & df_turn.arrival.isin(s_final.id)]

    assert s_final[(s_final.Time < 0) | (s_final.Time > 1435)].shape[
               0] == 0, 'Time request invalid'

    df_turn_new = delete_invalid_turnaround(df_turn, s_final)

    return s_h, s_final, df_turn_new
