import pandas as pd
import multiprocessing as mp
from DictAndRanges.dicts_and_ranges import Parameters


def get_new_entrant_airport(airport: str, series: pd.DataFrame):
    ne = []
    for i in Parameters().days:
        c = series[(series.InitialDate <= i) & (series.FinalDate >= i)].groupby(['Airline'], as_index=False).count()[
            ['Airline', 'id']]
        c['airport'] = airport
        c['day'] = i
        c['new_entrant'] = c.id < 7
        ne.extend(c[['airport', 'Airline', 'day', 'new_entrant']].values.tolist())
    return ne


def get_new_entrant(series: pd.DataFrame):
    ap = series.airport.unique()

    with mp.Pool() as pool:
        res = pool.starmap(get_new_entrant_airport, [(a, series[series.airport == a]) for a in ap])
        ne = []
        for r in res:
            ne.extend(r)

    return pd.DataFrame(ne, columns=['airport', 'Airline', 'day', 'new_entrant'])
