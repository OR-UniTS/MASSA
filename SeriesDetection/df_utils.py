def fix_df(db_slot):
    db_slot.matched = db_slot.matched.apply(lambda m: int(m) if m != 'N' else -1)
    db_slot.loc[db_slot.Time == 1440, 'Time'] = 1435

    idxs = db_slot[~db_slot.Historic & (db_slot.matched != -1)].matched
    idxs = db_slot[db_slot.Historic & db_slot.id.isin(idxs)].id
    db_slot.loc[db_slot.matched.isin(idxs), 'Historic'] = True
    idxs = db_slot[db_slot.Historic & (db_slot.matched != -1)].matched
    idxs = db_slot[~db_slot.Historic & db_slot.id.isin(idxs)].id
    db_slot.loc[db_slot.id.isin(idxs), 'Historic'] = True

    # db_slot['match_id'] = -1
    # idxs = db_slot[db_slot.matched != -1].matched
    # match_id = dict(zip(idxs, range(len(idxs))))
    # db_slot.loc[db_slot.matched != -1, 'match_id'] = range(len(idxs))
    # db_slot.loc[db_slot.id.isin(idxs), 'match_id'] = db_slot[db_slot.id.isin(idxs)].id.apply(lambda m: match_id[m])
    db_slot['HistoricChanged'] = False
    db_slot['HistoricOriginalTime'] = -1
    db_slot['HistoricOriginalSlot'] = -1

    return db_slot


def get_subset(db_slot, airports):
    df = db_slot[db_slot.airport.isin(airports)]
    idxs_not_included = db_slot[~db_slot.airport.isin(airports)].id
    df.loc[df.matched.isin(idxs_not_included), 'matched'] = -1
    return df
