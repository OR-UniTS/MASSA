import copy
import time
import numpy as np
import gurobipy as gb
from gurobipy import GRB
import pandas as pd

from MASSA.capacity import get_hour_capacities, get_hour_interval_capacities, get_interval_capacities
from MASSA.capacity_dict import HourCapDict, IntervalCapDict, CapacityObject, HourIntervalCapDict
from DictAndRanges.dicts_and_ranges import Parameters, AirportParameters
from MASSA.series import Series


class SalModel:

    def __init__(self, series, series_historic, series_non_historic, df_turn, df_ne, airlines=None,
                 initial_solution=False, window=6, capacity_correction=0, model_idx=None, fairness=False,
                 save_logs=False):
        self.fairness = fairness
        self.model_idx = model_idx
        fair = '_f' if self.fairness else ''
        self.model_name = 'model_' + str(model_idx) + '_' + str(window) + fair if model_idx is not None else ''

        (self.total_time, self.time_constraint, self.vars_time, self.alloc_time, self.cap_time, self.cap_constr_time,
         self.match_time, self.turn_time, self.obj_time, self.solve_time, self.fair_constr_time) = \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        # df fixing
        self.airlines = airlines

        self.series = series
        self.series_historic = series_historic
        self.series_non_historic = series_non_historic
        self.df_turn = df_turn
        self.df_ne = df_ne

        self.window = window
        self.var_window = self.window * 2 + 1

        self.df_matched = series_non_historic[series_non_historic.matched != -1]

        print('num series', series_non_historic.shape[0])
        self.capacity_correction = capacity_correction
        self.g_capacity, self.d_capacity, self.a_capacity = None, None, None
        self.hi_g_capacity, self.hi_d_capacity, self.hi_a_capacity = None, None, None
        self.i_capacity = None

        self.model = gb.Model(self.model_name, gb.Env(params={'MemLimit': 92}))
        self.model.setParam('MIPGap', 1e-3)
        if self.model_name != '' and save_logs:
            self.model.setParam('LogFile', 'Results/Logs/' + self.model_name)
        self.x, self.y, self.z, self.r, self.c = None, None, None, None, None
        self.y_ser_id = None
        self.initial_solution = initial_solution

        self.airports = series_non_historic.airport.unique()
        df = pd.read_csv('data/airports.csv')
        self.p = Parameters()
        self.air_p = dict(zip(df.airport, [AirportParameters(slot_size=slot_size) for slot_size in df['slot size']]))

        # var dicts
        self.h_caps = dict(zip(self.airports, [HourCapDict(airport, self.air_p[airport]) for airport in self.airports]))
        self.hi_caps = dict(
            zip(self.airports, [HourIntervalCapDict(airport, self.air_p[airport]) for airport in self.airports]))
        self.i_caps = dict(
            zip(self.airports, [IntervalCapDict(airport, self.air_p[airport]) for airport in self.airports]))

        self.series_airport_dict = {}
        self.series_dict: dict[int, Series] = {}
        self.n_hist_change = self.series_non_historic[self.series_non_historic.HistoricChanged].shape[0]

        self.costs = None

    def compute_capacity(self):
        self.cap_time = time.time()
        self.g_capacity, self.d_capacity, self.a_capacity = get_hour_capacities(
            self.series, self.series_historic, parameters=self.air_p, correction=self.capacity_correction)
        self.hi_g_capacity, self.hi_d_capacity, self.hi_a_capacity = get_hour_interval_capacities(
            self.series, self.series_historic, parameters=self.air_p, correction=self.capacity_correction)
        self.i_capacity = get_interval_capacities(self.series, self.series_historic, parameters=self.air_p,
                                                  correction=self.capacity_correction)

        self.cap_time = time.time() - self.cap_time
        print('capacity computation time', self.cap_time)
        self.total_time += self.cap_time

    def get_capacity_object(self):
        capacity = CapacityObject(self.g_capacity, self.d_capacity, self.a_capacity,
                                  self.hi_g_capacity, self.hi_d_capacity, self.hi_a_capacity, self.i_capacity)
        return copy.deepcopy(capacity)

    def set_precomputed_capacity(self, capacity: CapacityObject):
        self.g_capacity, self.d_capacity, self.a_capacity = capacity.g_capacity, capacity.d_capacity, capacity.a_capacity
        self.hi_g_capacity, self.hi_d_capacity, self.hi_a_capacity = capacity.hi_g_capacity, capacity.hi_d_capacity, capacity.hi_a_capacity
        self.i_capacity = capacity.i_capacity

    def add_vars(self):
        self.vars_time = time.time()
        self.y = self.model.addMVar(self.series_non_historic.shape[0], vtype=gb.GRB.BINARY)
        self.x = self.model.addMVar((self.series_non_historic.shape[0] * self.var_window + self.n_hist_change, 30),
                                    vtype=gb.GRB.BINARY)
        self.z = self.model.addMVar((self.df_matched.shape[0], 30), vtype=gb.GRB.INTEGER,
                                    ub=self.p.max_off_block_delay // self.p.time_unit_size)
        self.model.update()

        self.y_ser_id = np.zeros(self.series_non_historic.shape[0], dtype=int)
        x_idx, y_idx = 0, 0
        for i, airport in enumerate(self.airports):
            df_airport = self.series_non_historic[self.series_non_historic.airport == airport]
            self.series_airport_dict[airport] = []

            for _, ser in df_airport.iterrows():
                self.y_ser_id[y_idx] = ser.id
                s = Series(ser, self.h_caps[airport], self.hi_caps[airport], self.i_caps[airport], x_idx, self.y[y_idx],
                           y_idx, self.air_p[airport], self.window)
                self.series_dict[int(ser.id)] = s
                self.series_airport_dict[airport].append(s)
                x_idx = s.len_x_vars()
                y_idx += 1

        self.vars_time = time.time() - self.vars_time
        print('variables', self.vars_time)
        self.total_time += self.vars_time

    def add_allocation_constraints(self):
        self.alloc_time = time.time()
        self.time_constraint = time.time()

        for i, airport in enumerate(self.airports):
            for ser in self.series_airport_dict[airport]:
                x_ser = self.x[ser.idx: ser.idx + ser.width_series]
                self.model.addConstr((x_ser * ser.mask).sum() == ser.len_series * (1 - ser.y),
                                     name='alloc_x_y_' + str(ser.id))
                # self.model.addConstr(x_ser.sum(axis=1) >= x_ser[:, ser.InitialDate]*ser.len_series)

                # noinspection PyTypeChecker
                self.model.addConstr(x_ser.sum(axis=0) <= ser.mask[0], name='alloc_x' + str(ser.id))
                if ser.hist_change:
                    self.model.addConstr(ser.y == 0, name='y_' + str(ser.id))
                    # noinspection PyTypeChecker
                    self.model.addConstr(x_ser[-1].sum() == ser.len_series * x_ser[-1, ser.InitialDate],
                                         name='alloc_x_hist_' + str(ser.id))

        self.alloc_time = time.time() - self.alloc_time
        print('allocation constraint ', self.alloc_time)
        self.total_time += self.alloc_time

    def add_capacity_constraints(self):
        self.cap_constr_time = time.time()
        for i, airport in enumerate(self.airports):
            # print(i, airport, ' series:', self.series_non_historic[self.series_non_historic.airport == airport].shape[0])
            h_cap_airport = self.h_caps[airport]
            h_cap_airport: HourCapDict
            for hour in self.p.hours:
                if h_cap_airport.vars_idxs[hour]:
                    # noinspection PyTypeChecker
                    self.model.addConstr(
                        self.x[h_cap_airport.vars_idxs[hour]].sum(axis=0) <= self.g_capacity[airport][hour],
                        name='g_' + airport + '_' + str(hour))

                if h_cap_airport.dep_vars_idxs[hour]:
                    # noinspection PyTypeChecker
                    self.model.addConstr(
                        self.x[h_cap_airport.dep_vars_idxs[hour]].sum(axis=0) <= self.d_capacity[airport][hour],
                        name='d_' + airport + '_' + str(hour))
                if h_cap_airport.arr_vars_idxs[hour]:
                    # noinspection PyTypeChecker
                    self.model.addConstr(
                        self.x[h_cap_airport.arr_vars_idxs[hour]].sum(axis=0) <= self.a_capacity[airport][hour],
                        name='a_' + airport + '_' + str(hour))

            i_cap_airport = self.i_caps[airport]
            i_cap_airport: IntervalCapDict
            for slot in self.p.time_units:
                if i_cap_airport.vars_idxs[slot]:
                    # noinspection PyTypeChecker
                    self.model.addConstr(
                        self.x[i_cap_airport.vars_idxs[slot]].sum(axis=0) <= self.i_capacity[airport][slot],
                        name='i_' + airport + '_' + str(slot))

            hi_cap_airport = self.hi_caps[airport]
            hi_cap_airport: HourIntervalCapDict
            for slot in self.p.time_units:
                if hi_cap_airport.vars_idxs[slot]:
                    # noinspection PyTypeChecker
                    self.model.addConstr(
                        self.x[hi_cap_airport.vars_idxs[slot]].sum(axis=0) <= self.hi_g_capacity[airport][slot],
                        name='hig_' + airport + '_' + str(slot))
                if hi_cap_airport.dep_vars_idxs[slot]:
                    # noinspection PyTypeChecker
                    self.model.addConstr(
                        self.x[hi_cap_airport.dep_vars_idxs[slot]].sum(axis=0) <= self.hi_d_capacity[airport][slot],
                        name='hid_' + airport + '_' + str(slot))
                if hi_cap_airport.arr_vars_idxs[slot]:
                    # noinspection PyTypeChecker
                    self.model.addConstr(
                        self.x[hi_cap_airport.arr_vars_idxs[slot]].sum(axis=0) <= self.hi_a_capacity[airport][slot],
                        name='hia_' + airport + '_' + str(slot))

        self.cap_constr_time = time.time() - self.cap_constr_time
        print('capacity constraint ', self.cap_constr_time)
        self.total_time += self.cap_constr_time

    def add_match_constraints(self):
        self.match_time = time.time()
        matched = np.array([self.df_matched.id, self.df_matched.matched]).T

        min_shift = self.p.max_off_block_delay // self.p.time_unit_size
        for i, couple in enumerate(matched):
            sd, sa = self.series_dict[couple[1]], self.series_dict[couple[0]]
            if sd.hist_change != sa.hist_change:
                print('problem')
            # noinspection PyTypeChecker
            self.model.addConstr(sd.y == sa.y, name='match_' + str(sd.id) + '_' + str(sa.id))

            # arrival determines latest off block time, if dep series width is shorter than arrival, arrival dominates

            back_shift = sd.back_shift
            for_shift = sd.for_shift

            dep_ser = self.x[sd.idx_slot_requested - back_shift: sd.idx_slot_requested + for_shift]
            arr_ser = self.x[sa.idx_slot_requested - back_shift: sa.idx_slot_requested + for_shift]

            off_block_time = np.repeat(np.expand_dims(range(back_shift + for_shift), axis=1), repeats=30,
                                       axis=1) + min_shift + 1
            # noinspection PyTypeChecker
            self.model.addConstr((arr_ser * off_block_time - dep_ser * off_block_time).sum(axis=0) <= self.z[i],
                                 name='match 1_' + str(sd.id) + '_' + str(sa.id))
            # noinspection PyTypeChecker
            self.model.addConstr((arr_ser * off_block_time - dep_ser * off_block_time).sum(axis=0) >= 0,
                                 name='match 2_' + str(sd.id) + '_' + str(sa.id))

        self.match_time = time.time() - self.match_time
        print('coupled constraint', self.match_time)
        self.total_time += self.match_time

    def add_turn_around_constraints(self):
        self.turn_time = time.time()
        for i, row in self.df_turn.iterrows():
            sd, sa = self.series_dict[row.departure], self.series_dict[row.arrival]

            turnaround_time = 90 if row.wide_body else 30

            dep_ser = self.x[sd.idx_slot_requested - sd.back_shift: sd.idx_slot_requested + sd.for_shift, row.day]
            arr_ser = self.x[sa.idx_slot_requested - sa.back_shift: sa.idx_slot_requested + sa.for_shift, row.day]

            dep_shift_time = np.array(range(-sd.back_shift, sd.for_shift)) * self.p.time_unit_size
            arr_shift_time = np.array(range(-sa.back_shift, sa.for_shift)) * self.p.time_unit_size
            max_shift_time = max(max(dep_shift_time), max(arr_shift_time))
            scale_factor = 10 * self.p.time_unit_size
            sa_hi_constr = self.x[sa.hist_slot_idx, row.day] * ((sa.hist_time + turnaround_time) / scale_factor) \
                if sa.hist_change else 0
            sd_hi_constr = self.x[sd.hist_slot_idx, row.day] * (sd.hist_time / scale_factor) if sd.hist_change else 0

            self.model.addConstr(
                (arr_ser * ((arr_shift_time + sa.time_requested + turnaround_time) / scale_factor)).sum(axis=0)
                - (dep_ser * ((dep_shift_time + sd.time_requested) / scale_factor)).sum(axis=0)
                + sa_hi_constr - sd_hi_constr <= (sd.y + sa.y)
                * ((max_shift_time + max(sa.time_requested, sd.time_requested) + turnaround_time) / scale_factor),
                name='turn ' + str(sd.id) + ' ' + str(sa.id) + '_' + str(row.day))

        self.turn_time = time.time() - self.turn_time
        print('turnaround constraint time', self.turn_time)
        self.total_time += self.turn_time

    def add_new_entrant_constraints(self):
        self.ne_time = time.time()
        for airport in self.airports:
            for d in self.p.days:
                ser = self.series_non_historic[(self.series_non_historic.airport == airport) &
                                               (self.series_non_historic.InitialDate <= d) &
                                               (self.series_non_historic.FinalDate >= d)]
                ser_len = ser.shape[0]
                if ser_len >= 7:
                    ne = self.df_ne[(self.df_ne.airport == airport) & (self.df_ne.day == d) & self.df_ne.new_entrant
                                    ].Airline
                    ser_ne = ser[~ser.HistoricChanged & (ser.Airline.isin(ne))].id
                    ser_nne = ser[~ser.HistoricChanged & (~ser.Airline.isin(ne))].id
                    if ser_ne.shape[0] / ser_len >= .5:
                        y_idxs = [self.series_dict[id].y_idx for id in ser_ne]
                        self.model.addConstr(self.y[y_idxs].sum() <= .5 * ser_ne.shape[0],
                                             name='ne_' + str(airport) + '_d' + str(d))
                    if ser_nne.shape[0] / ser_len >= .5:
                        y_idxs = [self.series_dict[id].y_idx for id in ser_nne]
                        self.model.addConstr(self.y[y_idxs].sum() <= .5 * ser_nne.shape[0],
                                             name='nne_' + str(airport) + '_d' + str(d))

        self.ne_time = time.time() - self.ne_time
        print('new entrant constraint time', self.ne_time)
        self.total_time += self.ne_time

    def set_obj(self):
        self.obj_time = time.time()
        costs = np.zeros(self.x.shape, dtype=int)
        for ser in self.series_dict.values():
            costs[ser.idx: ser.idx + ser.width_series] = ser.costs
        self.costs = costs

        self.model.setObjective(
            (self.y * 30_000).sum() + (self.x * costs).sum() + (self.p.time_unit_size * self.z).sum())

        self.obj_time = time.time() - self.obj_time
        print('obj setting time', self.obj_time)
        self.total_time += self.obj_time

    def optimise(self):
        self.solve_time = time.time()
        self.model.optimize()

        self.solve_time = time.time() - self.solve_time
        print('optimisation time', self.solve_time)
        self.total_time += self.solve_time
        print('total_time', self.total_time)

    def print_solution(self):
        alloc = []
        non_alloc = []
        for airport in self.airports:
            for s in self.series_airport_dict[airport]:
                if s.y.x > 0.5:
                    non_alloc.append(s.idx)
                else:
                    alloc.append(s.idx)
        print('not allocated', len(non_alloc))
        print('allocated', len(alloc))

    def solve(self, y=None, allocated=None, hist_unchanged=None, capacity=None):
        t = time.time()
        if capacity is None:
            self.compute_capacity()
        else:
            self.set_precomputed_capacity(capacity)
        self.add_vars()
        self.add_allocation_constraints()
        self.add_capacity_constraints()
        self.add_match_constraints()
        self.add_turn_around_constraints()
        self.add_new_entrant_constraints()
        if y is not None:
            self.init_solution(y, allocated, hist_unchanged)
        self.time_constraint = time.time() - t
        print('constraint', self.time_constraint)
        self.set_obj()
        self.optimise()
        return self.model.status == GRB.OPTIMAL

    def solve_fairness(self, y, x, z, capacity: CapacityObject):
        t = time.time()
        self.airlines = self.series_non_historic.Airline.unique()
        self.add_vars()
        self.r = self.model.addMVar(1, vtype=gb.GRB.CONTINUOUS)
        # self.c = self.model.addMVar(1, vtype=gb.GRB.CONTINUOUS)
        self.add_allocation_constraints()
        self.set_precomputed_capacity(capacity)
        self.add_capacity_constraints()
        self.add_match_constraints()
        self.add_turn_around_constraints()
        self.add_new_entrant_constraints()

        fair_constr_time = time.time()
        costs = np.zeros(self.x.shape, dtype=int)

        match_id = dict(zip(self.df_matched.id, range(self.df_matched.shape[0])))
        for ser in self.series_dict.values():
            costs[ser.idx: ser.idx + ser.width_series] = ser.costs

        r = np.zeros(self.airlines.shape[0])

        fair_y_cost = np.ones(self.series_non_historic.shape[0]) * 30_000
        fair_x_costs = copy.deepcopy(costs)
        fair_z_costs = np.ones((self.df_matched.shape[0], 30)) * self.p.time_unit_size

        for j, airline in enumerate(self.airlines):
            df_airline = self.series_non_historic[self.series_non_historic.Airline == airline]
            df_hist_airline = self.series_historic[self.series_historic.Airline == airline]
            y_idxs = [self.series_dict[i].y_idx for i in df_airline.id]
            x_idxs = np.concatenate(
                [range(self.series_dict[i].idx, self.series_dict[i].idx + self.series_dict[i].width_series) for i in
                 df_airline.id])
            z_idxs = [match_id[i] for i in df_airline.id if i in match_id.keys()]
            n_requested = df_hist_airline.shape[0] + df_airline.shape[0]
            if len(z_idxs) > 0:
                self.model.addConstr(
                    self.y[y_idxs].sum() + (costs[x_idxs] * self.x[x_idxs]).sum() +
                    (self.p.time_unit_size * self.z[z_idxs]).sum() <= self.r * n_requested, name='cost_' + airline)
            else:
                self.model.addConstr(
                    self.y[y_idxs].sum() + (costs[x_idxs] * self.x[x_idxs]).sum() <= self.r * n_requested,
                    name='cost_' + airline)
            r[j] = (y[y_idxs].sum() + (costs[x_idxs] * x[x_idxs]).sum() + (self.p.time_unit_size * z[z_idxs]).sum()
                    / n_requested)

            fair_y_cost[y_idxs] /= n_requested
            fair_x_costs[x_idxs] = fair_x_costs[x_idxs] / n_requested
            fair_z_costs[z_idxs] /= n_requested

        self.fair_constr_time = time.time() - fair_constr_time
        self.time_constraint = time.time() - t
        print('constraint', self.time_constraint)
        self.y.Start = y
        self.x.Start = x
        self.z.Start = z
        self.r.Start = r.max()

        obj_time = time.time()
        self.model.setObjective(
            self.r + ((fair_y_cost * self.y).sum() + (fair_x_costs * self.x).sum() + (fair_z_costs * self.z).sum())
            / self.airlines.shape[0])
        self.obj_time = time.time() - obj_time
        print('obj setting time', self.obj_time)
        self.optimise()
        return self.model.status == GRB.OPTIMAL

    def get_solution(self):
        return copy.deepcopy(self.y.x), copy.deepcopy(self.x.x), copy.deepcopy(self.z.x), self.model.objVal

    def get_init_solution(self):
        allocated = self.y_ser_id[np.where(self.y.x < 0.5)[0]]
        hist_unchanged = [i for i in allocated if
                          self.series_dict[i].hist_change and self.x[self.series_dict[i].hist_slot_idx].x.sum() > 4]
        return copy.deepcopy(self.y.x), allocated, hist_unchanged, self.get_capacity_object()

    def delete_model(self):
        self.model.dispose()

    def assign_series(self):
        t = time.time()
        for _, s in self.series_non_historic.iterrows():
            ser = self.series_dict[s.id]
            if self.y[ser.y_idx].x == 0:
                ser.assigned = True
                ser.assigned_slot = [np.where(self.x[ser.idx: ser.idx + ser.width_series, i].x)[0][0]  # + ser.win_start
                                     for i in range(ser.InitialDate, ser.FinalDate + 1)]
                end = ser.for_shift + ser.hist_change
                times = np.array(range(-ser.back_shift, end)) * 5 + ser.time_requested
                if ser.hist_change:
                    times[-1] = ser.hist_time

                ser.assigned_time = np.array([times[i] for i in ser.assigned_slot])
                if ser.hist_change:
                    if self.x[ser.idx: ser.idx + ser.width_series - 1].x.sum() > 0.5 and self.x[
                        ser.hist_slot_idx].x.sum() > 0.5:
                        raise Exception("Invalid hist assignment", ser.id, '\n',
                                        self.x[ser.idx: ser.idx + ser.width_series - 1].x)
        z_idx = 0
        for _, s in self.df_matched.iterrows():
            arr_ser = self.series_dict[s.id]
            if self.y[arr_ser.y_idx].x == 0:
                ser = self.series_dict[s.matched]
                block_time_requested = arr_ser.time_requested - ser.time_requested
                block_time_assigned = arr_ser.assigned_time - ser.assigned_time
                off_block_tol = np.abs(block_time_assigned - block_time_requested)
                invalid = off_block_tol[off_block_tol > self.p.max_off_block_delay]
                if invalid.shape[0] > 0:
                    dep_ser = self.x[ser.idx_slot_requested - ser.back_shift: ser.idx_slot_requested + ser.for_shift].x
                    arr_ser_v = self.x[
                                arr_ser.idx_slot_requested - ser.back_shift: arr_ser.idx_slot_requested + ser.for_shift].x
                    min_shift = self.p.max_off_block_delay // self.p.time_unit_size
                    off_block_time = (
                            np.repeat(np.expand_dims(range(ser.back_shift + ser.for_shift), axis=1), repeats=30,
                                      axis=1) + min_shift + 1)

                    print((arr_ser_v * off_block_time - dep_ser * off_block_time).sum(axis=0))
                    print(self.z[z_idx].x)
                    print(dep_ser)
                    print(arr_ser_v)

                    print(block_time_assigned, block_time_requested)
                    print((block_time_assigned - block_time_requested))
                    print(ser.time_requested, arr_ser.time_requested)
                    print(ser.assigned_time)
                    print(arr_ser.assigned_time)
                    raise Exception("Invalid off block", ser.id, arr_ser.id,
                                    np.where(off_block_tol > self.p.max_off_block_delay),
                                    self.x[ser.idx: ser.idx + ser.width_series].x, '\n',
                                    self.x[arr_ser.idx: arr_ser.idx + arr_ser.width_series].x)
            z_idx += 1

        for _, row in self.df_turn.iterrows():
            sd, sa = self.series_dict[row.departure], self.series_dict[row.arrival]
            if self.y[sd.y_idx].x == 0 and self.y[sa.y_idx].x == 0:
                day = row.day
                turnaround_time = 90 if row.wide_body else 30
                sd_days = dict(zip(range(sd.InitialDate, sd.FinalDate + 1), range(sd.len_series)))
                sa_days = dict(zip(range(sa.InitialDate, sa.FinalDate + 1), range(sa.len_series)))
                turn_around_tol = sd.assigned_time[sd_days[day]] - sa.assigned_time[sa_days[day]] - turnaround_time
                if turn_around_tol < 0:
                    print(sa.assigned_time[sa_days[day]])
                    print(sd.assigned_time[sd_days[day]])
                    print(turnaround_time)
                    raise Exception("Invalid turn around", sd.id, sa.id, np.where(turn_around_tol < 0), turnaround_time,
                                    self.x[sd.idx: sd.idx + sd.width_series, day].x, '\n',
                                    self.x[sa.idx: sa.idx + sa.width_series, day].x)
        print('assignment time', time.time() - t)

    def init_solution(self, y, allocated, hist_unchanged):
        x = np.zeros((self.series_non_historic.shape[0] * self.var_window + self.n_hist_change, 30))
        z = np.zeros((self.df_matched.shape[0], 30))

        print('allocated', len(allocated), 'hist changed', len(hist_unchanged))

        for s in allocated:
            ser = self.series_dict[s]
            if s not in hist_unchanged:
                x[ser.idx_slot_requested, ser.InitialDate: ser.FinalDate + 1] = 1
            else:
                x[ser.hist_slot_idx, ser.InitialDate: ser.FinalDate + 1] = 1

        self.y.Start = y
        self.x.Start = x
        self.z.Start = z

    def make_solution_df(self, save):
        self.assign_series()
        df_series = self.series_non_historic.copy(deep=True)
        idxs = [self.series_dict[i].y_idx for i in df_series.id]
        df_series['y'] = self.y.x[idxs]

        x = - np.ones((df_series.shape[0], 30), dtype=int)
        for s in df_series.id:
            ser = self.series_dict[s]
            if ser.assigned:
                x[ser.y_idx, ser.InitialDate: ser.FinalDate + 1] = ser.assigned_time

        x = x[idxs]
        for i in range(30):
            df_series[str(i)] = x[:, i]

        if save:
            df_series.to_csv('Results/' + self.model_name + '.csv', index_label=False, index=False)

        df_times = pd.DataFrame({'model': [self.model_idx], 'win': [self.window], 'fair': [self.fairness],
                                 'total': [self.total_time], 'sol': [self.solve_time], 'capacity': [self.cap_time],
                                 'constr': [self.time_constraint], 'obj': [self.obj_time], 'vars': [self.vars_time],
                                 'alloc': [self.alloc_time], 'cap_constr': [self.cap_constr_time],
                                 'match': [self.match_time], 'turn': [self.turn_time],
                                 'fair_time': [self.fair_constr_time], 'opt_time': [self.model.Runtime]})
        df_times.to_csv('Results/' + self.model_name + '_time.csv', index_label=False, index=False)
