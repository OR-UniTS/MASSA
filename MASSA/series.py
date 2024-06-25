import numpy as np

from MASSA.capacity_dict import HourCapDict, IntervalCapDict, HourIntervalCapDict
from DictAndRanges.dicts_and_ranges import Parameters, AirportParameters


class Series:
    def __init__(self, ser, h_caps: HourCapDict, hi_caps: HourIntervalCapDict, i_caps: IntervalCapDict, idx, y, y_idx, p: AirportParameters,
                 window=6):
        self.idx = idx
        self.id = ser.id
        self.airport = ser.airport
        self.airline = ser.Airline
        self.y_idx = y_idx
        self.y = y
        self.time_requested = ser.Time
        self.slot_requested = ser.Slot
        self.flow = ser.Flow
        self.hist_change = ser.HistoricChanged
        self.hist_time = ser.HistoricOriginalTime
        self.hist_slot = ser.HistoricOriginalSlot

        self.assigned = False
        self.assigned_slot = None
        self.assigned_time = None

        self.len_series = ser.FinalDate - ser.InitialDate + 1
        self.InitialDate, self.FinalDate = ser.InitialDate, ser.FinalDate

        self.window = window
        if ser.matched_slot == -1:
            self.back_shift, self.for_shift = min(self.slot_requested, self.window), min(self.window, 287 - self.slot_requested) + 1
        else:
            self.back_shift = min(self.slot_requested, ser.matched_slot, self.window)
            self.for_shift = min(self.window, 287 - self.slot_requested, 287 - ser.matched_slot) + 1

        # self.win_start, self.win_end = min(self.slot_requested, self.window), min(self.window, 287 - self.slot_requested) + 1
        start, end = max(0, self.slot_requested - self.back_shift), min(287, self.slot_requested + self.for_shift - 1)
        self.width_series = end - start + self.hist_change + 1

        self.costs = np.abs(
            np.repeat(
                np.expand_dims(np.array(range(-self.back_shift, self.for_shift)), axis=1), 30, axis=1)) * p.time_unit_size
        if self.hist_change:
            self.costs = np.vstack([self.costs, np.ones(30) * np.abs(self.hist_time - self.time_requested)])

        self.idx_slot_requested = idx + self.back_shift
        self.hist_slot_idx = idx + self.width_series - 1

        for i, slot in enumerate(range(start, end + 1)):
            for j in p.unit_to_hours[slot]:
                h_caps.vars_idxs[j].append(self.idx + i)
            for j in p.unit_to_intervals[slot]:
                i_caps.vars_idxs[j].append(self.idx + i)
            for j in p.unit_to_hour_intervals[slot]:
                hi_caps.vars_idxs[j].append(self.idx + i)

        if self.flow == 'D':
            for i, slot in enumerate(range(start, end + 1)):
                for j in p.unit_to_hours[slot]:
                    h_caps.dep_vars_idxs[j].append(self.idx + i)
                for j in p.unit_to_hour_intervals[slot]:
                    hi_caps.dep_vars_idxs[j].append(self.idx + i)
        else:
            for i, slot in enumerate(range(start, end + 1)):
                for j in p.unit_to_hours[slot]:
                    h_caps.arr_vars_idxs[j].append(self.idx + i)
                for j in p.unit_to_hour_intervals[slot]:
                    hi_caps.arr_vars_idxs[j].append(self.idx + i)

        if self.hist_change:
            for j in p.unit_to_hours[self.hist_slot]:
                h_caps.vars_idxs[j].append(self.hist_slot_idx)
            for j in p.unit_to_intervals[self.hist_slot]:
                i_caps.vars_idxs[j].append(self.hist_slot_idx)
            for j in p.unit_to_hour_intervals[self.hist_slot]:
                hi_caps.vars_idxs[j].append(self.hist_slot_idx)

            if self.flow == 'D':
                for j in p.unit_to_hours[self.hist_slot]:
                    h_caps.dep_vars_idxs[j].append(self.hist_slot_idx)
                for j in p.unit_to_hour_intervals[self.hist_slot]:
                    hi_caps.dep_vars_idxs[j].append(self.hist_slot_idx)
            else:
                for j in p.unit_to_hours[self.hist_slot]:
                    h_caps.arr_vars_idxs[j].append(self.hist_slot_idx)
                for j in p.unit_to_hour_intervals[self.hist_slot]:
                    hi_caps.arr_vars_idxs[j].append(self.hist_slot_idx)

        self.mask = np.zeros((self.width_series, 30), dtype=bool)
        self.mask[:, ser.InitialDate: ser.FinalDate + 1] = True

    def len_x_vars(self):
        return self.idx + self.width_series

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)