import random
import time
import numpy as np
import pandas as pd
import gurobipy as gb
import os

from SeriesDetection.df_utils import get_subset
from DictAndRanges.dicts_and_ranges import Parameters
from MASSA.sal_model import SalModel
from DataGeneration.series_generation import generate_series
from DataGeneration.turnaround import get_turnaround
from DataGeneration.new_entrant import get_new_entrant

SAVE_RESULTS = True
PERCENTAGE_NEW = 0.05
PERCENTAGE_HRC = 0.32
RUNS = 30
parameters = Parameters()
WIN_SIZE = 30 // parameters.time_unit_size

total_time = time.time()

db_slot = pd.read_csv('data/friday_2019.csv')
airlines = db_slot.Airline.unique()

# airports_considered = db_slot.airport.unique()[28:34]
airports_considered = db_slot.airport.unique()
print('Airport considered:',
      dict(zip(airports_considered, [db_slot[db_slot.airport == a].shape[0] for a in airports_considered])))

series = get_subset(db_slot, airports_considered)

# get_turnaround(series).to_csv('data/turnaround_processed.csv', index=False)
df_turnaround = pd.read_csv('data/turnaround_processed.csv')

series_historic, series_non_historic = (series[series.Historic], series[~series.Historic])

i = 0
while i < RUNS:
    random.seed(i)
    np.random.seed(i)
    print('\n\n\niteration', i, '********************************************\n')

    s_historic, s_non_historic, df_turn = generate_series(series_historic.copy(deep=True),
                                                          series_non_historic.copy(deep=True), airlines=airlines,
                                                          start_idx_new_series=series.id.max() + 1,
                                                          df_turn=df_turnaround.copy(deep=True),
                                                          percentage=PERCENTAGE_NEW, percentage_hrc=PERCENTAGE_HRC)
    df_ne = get_new_entrant(s_non_historic)

    print('***** WIN 0\n')
    logfile = 'Results/Logs/model_' + str(i) + '_0'
    if not os.path.isdir('Results/Logs'):
        os.makedirs('Results/Logs')
    if os.path.exists(logfile):
        os.remove(logfile)

    model = SalModel(series=series, series_historic=s_historic, series_non_historic=s_non_historic,
                     df_turn=df_turn, df_ne=df_ne, model_idx=i, fairness=False, window=0, save_logs=SAVE_RESULTS)
    optimal = model.solve()
    if optimal:
        model.print_solution()
        model.make_solution_df(save=SAVE_RESULTS)
        y, allocated, hist_unchanged, capacity = model.get_init_solution()
    model.delete_model()

    print('\n***** WIN', WIN_SIZE, '\n')
    logfile = 'Results/Logs/model_' + str(i) + '_' + str(WIN_SIZE)
    if os.path.exists(logfile):
        os.remove(logfile)

    model = SalModel(series=series, series_historic=s_historic, series_non_historic=s_non_historic,
                     df_turn=df_turn, df_ne=df_ne, model_idx=i, fairness=False, window=WIN_SIZE, save_logs=SAVE_RESULTS)
    try:
        if optimal:
            optimal = model.solve(y, allocated, hist_unchanged, capacity)
        else:
            optimal = model.solve()
        if optimal:
            model.print_solution()
            model.make_solution_df(save=SAVE_RESULTS)
            capacity = model.get_capacity_object()
            y, x, z, obj_val = model.get_solution()

    except gb.GurobiError:
        print(i, 'WIN', WIN_SIZE, 'skipped, out of memory')
        optimal = False
        model.print_solution()
        model.make_solution_df(save=SAVE_RESULTS)
    model.delete_model()

    print('\n***** FAIR\n')
    logfile = 'Results/Logs/model_' + str(i) + '_' + str(WIN_SIZE) + '_f'
    if os.path.exists(logfile):
        os.remove(logfile)

    model = SalModel(series=series, series_historic=s_historic, series_non_historic=s_non_historic,
                     df_turn=df_turn, df_ne=df_ne, model_idx=i, fairness=True, window=WIN_SIZE, save_logs=SAVE_RESULTS)
    try:
        optimal = model.solve_fairness(y, x, z, capacity)
        if optimal:
            model.print_solution()
            model.make_solution_df(save=SAVE_RESULTS)
    except gb.GurobiError:
        print(i, 'FAIR skipped, out of memory')
        model.print_solution()
        model.make_solution_df(save=SAVE_RESULTS)
    model.delete_model()

    i += 1
