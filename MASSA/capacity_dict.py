from DictAndRanges.dicts_and_ranges import Parameters


class HourCapDict:
    def __init__(self, airport, parameters: Parameters):
        self.airport = airport
        self.vars_idxs = dict(zip(parameters.hours, [[] for _ in parameters.hours]))
        self.dep_vars_idxs = dict(zip(parameters.hours, [[] for _ in parameters.hours]))
        self.arr_vars_idxs = dict(zip(parameters.hours, [[] for _ in parameters.hours]))


class HourIntervalCapDict:
    def __init__(self, airport, parameters: Parameters):
        self.airport = airport
        self.vars_idxs = dict(zip(parameters.time_units, [[] for _ in parameters.time_units]))
        self.dep_vars_idxs = dict(zip(parameters.time_units, [[] for _ in parameters.time_units]))
        self.arr_vars_idxs = dict(zip(parameters.time_units, [[] for _ in parameters.time_units]))


class IntervalCapDict:
    def __init__(self, airport, parameters: Parameters):
        self.airport = airport
        self.vars_idxs = dict(zip(parameters.time_units, [[] for _ in parameters.time_units]))


class CapacityObject:
    def __init__(self, g_capacity, d_capacity, a_capacity, hi_g_capacity, hi_d_capacity, hi_a_capacity, i_capacity):
        self.g_capacity, self.d_capacity, self.a_capacity = g_capacity, d_capacity, a_capacity
        self.hi_g_capacity, self.hi_d_capacity, self.hi_a_capacity = hi_g_capacity, hi_d_capacity, hi_a_capacity
        self.i_capacity = i_capacity

