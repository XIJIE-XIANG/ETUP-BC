import cv2
import zipfile
import numpy as np
import pandas as pd
from .timers import Timer
from os.path import splitext


class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        self.iterator = pd.read_csv(path_to_event_file, delim_whitespace=True, header=None,
                                    names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c',
                                    skiprows=start_index + 1, chunksize=num_events, nrows=None, memory_map=True)

    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            event_window = self.iterator.__next__().values
            # event_window = event_window[:, [1, 2, 3, 0]]
            # event_window[:, -1] -= event_window[0, -1]
            # event_window[:, -1] *= 1e6
        return event_window #.astype(np.int32)


class FixedDurationEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, duration_us=50000.0, start_index=0):
        print('Will use fixed duration event windows of size {:.2f} us'.format(duration_us))
        print('Output frame rate: {:.1f} Hz'.format(1e6 / duration_us))
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt', '.zip'])
        self.is_zip_file = (file_extension == '.zip')

        if self.is_zip_file:  # '.zip'
            self.zip_file = zipfile.ZipFile(path_to_event_file)
            files_in_archive = self.zip_file.namelist()
            assert(len(files_in_archive) == 1)  # make sure there is only one text file in the archive
            self.event_file = self.zip_file.open(files_in_archive[0], 'r')
        else:
            self.event_file = open(path_to_event_file, 'r')

        # ignore header + the first start_index lines
        for i in range(1 + start_index):
            self.event_file.readline()

        self.last_stamp = None
        self.duration_s = duration_us / 1e6

    def __iter__(self):
        return self

    def __del__(self):
        if self.is_zip_file:
            self.zip_file.close()

        self.event_file.close()

    def __next__(self):
        with Timer('Reading event window from file'):
            event_list = []
            for line in self.event_file:
                if self.is_zip_file:
                    line = line.decode("utf-8")
                t, x, y, pol = line.split(' ')
                t, x, y, pol = float(t), int(x), int(y), int(pol)
                # event_list.append([x, y, pol, t])
                event_list.append([t, x, y, pol])
                if self.last_stamp is None:
                    self.last_stamp = t
                if t > self.last_stamp + self.duration_s:
                    self.last_stamp = t
                    event_window = np.array(event_list)
                    # event_window[:, -1] -= event_window[0, -1]
                    # event_window[:, -1] *= 1e6
                    return event_window #.astype(np.int32)

        raise StopIteration


def FixedDurationEventReaderNPZ(path_to_event_file, duration_us=50.0, rangeX=970, rangeY=625):
    events = np.load(path_to_event_file)
    x = (events['x'] / 32).astype(np.float32)
    y = (events['y'] / 32).astype(np.float32)
    p = np.where(events['polarity'] == 0, -1, 1).astype(np.int8)
    t = events['timestamp']
    idx = np.where((x < rangeX) & (y < rangeY))[0]
    x, y, p, t = x[idx], y[idx], p[idx], t[idx]

    last_t = t[0]
    cnt_window = int((t[-1] - t[0]) / duration_us) + 1

    windows = []
    for c in range(cnt_window):
        idx = np.where((last_t <= t) & (t < last_t + duration_us))[0]
        x_, y_, p_, t_ = x[idx], y[idx], p[idx], t[idx]
        window = np.concatenate([np.expand_dims(x_, axis=1), np.expand_dims(y_, axis=1), np.expand_dims(p_, axis=1), np.expand_dims(t_, axis=1)], axis=1).astype(np.float32)
        window[:, -1] -= window[0, -1]
        windows.append(window)

        if idx[-1] == len(t) - 1:
            break
        last_t = t[idx[-1] + 1]

    return np.array(windows)


def FixedSizeEventReaderNPZ(path_to_event_file, duration_us=50.0, rangeX=970, rangeY=625):
    events = np.load(path_to_event_file)
    x = (events['x'] / 32).astype(np.float32)
    y = (events['y'] / 32).astype(np.float32)
    p = np.where(events['polarity'] == 0, -1, 1).astype(np.int8)
    t = events['timestamp']
    idx = np.where((x < rangeX) & (y < rangeY))[0]
    x, y, p, t = x[idx], y[idx], p[idx], t[idx]

    last_t = t[0]
    cnt_window = int((t[-1] - t[0]) / duration_us) + 1

    windows = []
    for c in range(cnt_window):
        idx = np.where((last_t <= t) & (t < last_t + duration_us))[0]
        x_, y_, p_, t_ = x[idx], y[idx], p[idx], t[idx]
        window = np.concatenate([np.expand_dims(t_, axis=1), np.expand_dims(x_, axis=1), np.expand_dims(y_, axis=1), np.expand_dims(p_, axis=1)], axis=1).astype(np.float32)
        windows.append(window)

        if idx[-1] == len(t) - 1:
            break
        last_t = t[idx[-1] + 1]

    return np.array(windows)


# events -> voxel
def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


# events -> 0-1 matrix
def events_to_0_1_matrix(events, B, rangeX, rangeY):
    x = events[:, 0]
    y = events[:, 1]
    p = events[:, 2]
    t = events[:, 3]

    dt = (t[-1] - t[0]) / B
    t_norm = t - t[0]
    bins = []
    for bi in range(B):
        if bi == B - 1:
            idx = np.where((bi * dt <= t_norm) & (t_norm <= (bi + 1) * dt))
        else:
            idx = np.where((bi * dt <= t_norm) & (t_norm < (bi + 1) * dt))
        bin = np.zeros((rangeY, rangeX))
        bin[y[idx], x[idx]] = p[idx]
        bins.append(bin)
    bins = np.stack(bins).astype(np.float32)
    return bins


# divide events by duration
def divide_duration(events, duration, factor, rangeX, rangeY):
    short_duration = duration // factor
    div_events = []
    events_2D = []
    for idx_t in range(factor):
        start_t = idx_t * short_duration
        end_t = (idx_t + 1) * short_duration
        if idx_t != factor - 1:
            idx = np.where((start_t <= events[:, -1]) & (events[:, -1] < end_t))
        else:
            idx = np.where((start_t <= events[:, -1]) & (events[:, -1] <= end_t))
        cur_events = events[idx[0], :]
        div_events.append(cur_events)
        cur_events_2D = np.zeros((rangeY, rangeX)).astype(np.float32)
        for i in range(cur_events.shape[0]):
            if cur_events[i, 2] > 0:
                cur_events_2D[cur_events[i, 1], cur_events[i, 0]] += 1
            else:
                cur_events_2D[cur_events[i, 1], cur_events[i, 0]] -= 1
        # cur_events_2D[cur_events[:, 1], cur_events[:, 0]] = np.where(cur_events[:, 2] > 0, cur_events_2D[cur_events[:, 1], cur_events[:, 0]] + 1, cur_events_2D[cur_events[:, 1], cur_events[:, 0]] - 1)
        events_2D.append(cur_events_2D)
    events_2D = np.stack(events_2D)
    return div_events, events_2D / np.max(np.abs(events_2D))

def divide_duration_(events, short_duration, bin=None):
    t1 = events[0, -1]
    t1_ = events[-1, -1]
    delta = t1_ - t1
    div_events = []
    if bin == None:
        factor = delta // short_duration
        for idx_t in range(int(factor)):
            start_t = t1 + idx_t * short_duration
            end_t = t1 + (idx_t + 1) * short_duration
            if idx_t != factor - 1:
                idx = np.where((start_t <= events[:, -1]) & (events[:, -1] < end_t))
            else:
                idx = np.where((start_t <= events[:, -1]) & (events[:, -1] <= end_t))
            cur_events = events[idx[0], :]
            div_events.append(cur_events)
    else:
        duration = delta / bin
        for idx_t in range(bin):
            start_t = t1 + idx_t * duration
            end_t = start_t + duration
            if idx_t != bin - 1:
                idx = np.where((start_t <= events[:, -1]) & (events[:, -1] < end_t))
            else:
                idx = np.where((start_t <= events[:, -1]) & (events[:, -1] <= end_t))
            cur_events = events[idx[0], :]
            div_events.append(cur_events)
    return div_events

def divide_duration_ON_OFF(events, duration, factor, rangeX, rangeY):
    short_duration = duration / factor
    div_events = []
    events_2D_ON = []
    events_2D_OFF = []
    for idx_t in range(factor):
        start_t = events[0, -1] + idx_t * short_duration
        end_t = start_t + short_duration
        if idx_t != factor - 1:
            idx = np.where((start_t <= events[:, -1]) & (events[:, -1] < end_t))
        else:
            idx = np.where((start_t <= events[:, -1]) & (events[:, -1] <= end_t))
        cur_events = events[idx[0], :]
        div_events.append(cur_events)
        cur_events_2D_ON = np.zeros((rangeY, rangeX)).astype(np.float32)
        cur_events_2D_OFF = np.zeros((rangeY, rangeX)).astype(np.float32)
        for i in range(cur_events.shape[0]):
            if cur_events[i, 2] > 0:
                cur_events_2D_ON[round(cur_events[i, 1]), round(cur_events[i, 0])] += 1
            else:
                cur_events_2D_OFF[round(cur_events[i, 1]), round(cur_events[i, 0])] -= 1
        events_2D_ON.append(cur_events_2D_ON)
        events_2D_OFF.append(cur_events_2D_OFF)
    events_2D_ON = np.stack(events_2D_ON)
    events_2D_OFF = np.stack(events_2D_OFF)
    return div_events, events_2D_ON, events_2D_OFF

# divide events by events count
def divide_count(events, count, factor, rangeX, rangeY):
    small_count = count // factor
    div_events = []
    events_2D = []
    for idx in range(factor):
        cur_events = events[idx * small_count : (idx + 1) * small_count, :]
        div_events.append(cur_events)
        cur_events_2D = np.zeros((rangeY, rangeX)).astype(np.float32)
        for i in range(cur_events.shape[0]):
            if cur_events[i, 2] > 0:
                cur_events_2D[cur_events[i, 1], cur_events[i, 0]] += 1
            else:
                cur_events_2D[cur_events[i, 1], cur_events[i, 0]] -= 1
        # cur_events_2D[cur_events[:, 1], cur_events[:, 0]] = np.where(cur_events[:, 2] > 0, 1, -1)
        events_2D.append(cur_events_2D)
    events_2D = np.stack(events_2D)
    return div_events, events_2D #/ np.max(np.abs(events_2D))
    # return div_events, np.stack(events_2D)

def divide_count_(events, count=10000):
    factor = events.shape[0] // count
    div_events = []
    for idx in range(int(factor)):
        cur_events = events[idx * count : (idx + 1) * count, :]
        div_events.append(cur_events)
    return div_events

def divide_count_ON_OFF(events, count, factor, rangeX, rangeY):
    small_count = count // factor
    div_events = []
    events_2D_ON = []
    events_2D_OFF = []
    for idx in range(factor):
        cur_events = events[idx * small_count : (idx + 1) * small_count, :]
        div_events.append(cur_events)
        cur_events_2D_ON = np.zeros((rangeY, rangeX)).astype(np.float32)
        cur_events_2D_OFF = np.zeros((rangeY, rangeX)).astype(np.float32)
        for i in range(cur_events.shape[0]):
            if cur_events[i, 2] > 0:
                cur_events_2D_ON[cur_events[i, 1], cur_events[i, 0]] += 1
            else:
                cur_events_2D_OFF[cur_events[i, 1], cur_events[i, 0]] -= 1
        # cur_events_2D[cur_events[:, 1], cur_events[:, 0]] = np.where(cur_events[:, 2] > 0, 1, -1)
        events_2D_ON.append(cur_events_2D_ON)
        events_2D_OFF.append(cur_events_2D_OFF)

    events_2D_ON = np.stack(events_2D_ON)
    events_2D_OFF = np.stack(events_2D_OFF)

    return div_events, events_2D_ON, events_2D_OFF #/ np.max(np.abs(events_2D))
    # return div_events, np.stack(events_2D)

def form_events(event_window):
    event_window = event_window[:, [1, 2, 3, 0]]
    event_window[:, 3] -= event_window[0, 3]
    event_window[:, 3] *= 1e6
    event_window = event_window.astype(np.int32)
    return event_window


def events_resize(events_2D, up=2, resize='cv2.INTER_CUBIC', thresh=0.5): # INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
    LR_events = events_2D[..., ::up, ::up]
    HR_events = np.zeros((LR_events.shape[-3], LR_events.shape[-2] * 2, LR_events.shape[-1] * 2))
    for k in range(LR_events.shape[-3]):
        HR_events[k, ...] = cv2.resize(LR_events[k, ...], (HR_events.shape[2], HR_events.shape[1]),
                                       interpolation=eval(resize))
        HR_events[k, ...] = np.where(np.abs(HR_events[k, ...]) > thresh, HR_events[k, ...], 0)
    return HR_events