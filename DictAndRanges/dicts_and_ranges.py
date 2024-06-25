import numpy as np


class Parameters:
    def __init__(self):
        self.days = range(30)
        self.time_unit_size = 5
        self.time_units = range(60 * 24 // self.time_unit_size)
        self.hours = range(24)
        self.max_off_block_delay = 15

        self.time_to_unit = dict(zip(range(0, 60 * 24, self.time_unit_size), self.time_units))
        self.unit_to_time = dict(zip(self.time_units, range(0, 60 * 24, self.time_unit_size)))

        self.time_to_hour = dict(zip(range(1440), [i // 60 for i in range(1440)]))


class AirportParameters(Parameters):
    def __init__(self, slot_size=15, interval_size_min=15):
        super().__init__()
        self.slot_size = slot_size

        self.unit_to_hours = dict(zip(range(1440 // self.time_unit_size),
                                      [np.unique([(i + j) // 12 for j in range(min(slot_size//self.time_unit_size, 288 - i))])
                                                                          for i in range(288)]))
        units_in_h = 60 // self.time_unit_size
        slot_units = self.slot_size//self.time_unit_size
        self.hour_to_units = dict(zip(range(24), [[j for j in range(i*units_in_h - slot_units + 1, (i + 1)*units_in_h) if j >= 0]
                                                  for i in range(24)]))

        self.interval_size = interval_size_min // self.time_unit_size
        self.unit_to_intervals = dict(
            zip(self.time_units, [[j for j in range(max(unit - self.interval_size + 1, 0),
                                                    min(unit + self.slot_size // self.time_unit_size, 1440 // self.time_unit_size))]
                                  for unit in self.time_units]))

        self.unit_to_hour_intervals = dict(
            zip(self.time_units, [[j for j in range(max(unit - 60 // self.time_unit_size + 1, 0),
                                                    min(unit + self.interval_size, 1440 // self.time_unit_size))]
                                  for unit in self.time_units]))
