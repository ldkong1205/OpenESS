import math
from typing import Dict, Tuple

import h5py
import hdf5plugin
from numba import jit
import numpy as np


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int, max_events_per_data: int = -1) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            print('Error', 'start', t_start_us, 'end', t_end_us)
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset

        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size

        return events

    def get_events_fixed_num(self, t_end_us: int, nr_events: int = 100000) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) with fixed number of events
        Parameters
        ----------
        t_end_us: end time in microseconds
        nr_events: number of events to load
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        t_end_us -= self.t_offset

        t_end_lower_ms, t_end_upper_ms = self.get_conservative_ms(t_end_us)
        t_end_lower_ms_idx = self.ms2idx(t_end_lower_ms)
        t_end_upper_ms_idx = self.ms2idx(t_end_upper_ms)

        if t_end_lower_ms_idx is None or t_end_upper_ms_idx is None:
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_end_lower_ms_idx:t_end_upper_ms_idx])
        _, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_end_us, t_end_us)
        t_end_us_idx = t_end_lower_ms_idx + idx_end_offset
        t_start_us_idx = t_end_us_idx - nr_events
        if t_start_us_idx < 0:
            t_start_us_idx = 0

        for dset_str in self.events.keys():
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])

        return events

    def get_events_fixed_num_recurrent(self, t_start_us_idx: int, t_end_us_idx: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) with fixed number of events
        Parameters
        ----------
        t_start_us_idx: start id
        t_end_us_idx: end id
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us_idx < t_end_us_idx

        events = dict()
        for dset_str in self.events.keys():
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])

        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    def get_conservative_ms(ts_us: int) -> Tuple[int, int]:
        """Convert time in microseconds into milliseconds
        ----------
        ts_us:    time in microseconds
        Returns
        -------
        ts_lower_ms:    lower millisecond
        ts_upper_ms:    upper millisecond
        """
        ts_lower_ms = math.floor(ts_us / 1000)
        ts_upper_ms = math.ceil(ts_us / 1000)
        return ts_lower_ms, ts_upper_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]