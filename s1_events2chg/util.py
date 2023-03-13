import os
import json
import time
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
from math import ceil, floor
from itertools import repeat
from torch.nn import ZeroPad2d
from collections import OrderedDict, defaultdict

def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def mean(l):
    return 0 if len(l) == 0 else sum(l) / len(l)


def quick_norm(img):
    return (img - torch.min(img))/(torch.max(img) - torch.min(img) + 1e-5)


def robust_min(img, p=5):
    return np.percentile(img.ravel(), p)


def robust_max(img, p=95):
    return np.percentile(img.ravel(), p)


def normalize(img, m=10, M=90):
    return np.clip((img - robust_min(img, m)) / (robust_max(img, M) - robust_min(img, m)), 0.0, 1.0)


def ffmpeg_glob_cmd(input_folder, output_path=None):
    if output_path is None:
        output_path = os.path.join(input_folder, 'a_video.mp4')
    return ['ffmpeg', '-y', '-pattern_type', 'glob', '-i',
            os.path.join(input_folder, '*.png'), '-framerate', '20',
            output_path]


def optimal_crop_size(max_size, max_subsample_factor, safety_margin=0):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    crop_size += safety_margin * pow(2, max_subsample_factor)
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders, safety_margin=0):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders, safety_margin)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders, safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ZeroPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)

    def crop(self, img):
        return img[..., self.iy0:self.iy1, self.ix0:self.ix1]


def format_power(size):
    power = 1e3
    n = 0
    power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]

def recursive_clone(tensor):
    """
    Assumes tensor is a torch.tensor with 'clone()' method, possibly
    inside nested iterable.
    E.g., tensor = [(pytorch_tensor, pytorch_tensor), ...]
    """
    if hasattr(tensor, 'clone'):
        return tensor.clone()
    try:
        return type(tensor)(recursive_clone(t) for t in tensor)
    except TypeError:
        print('{} is not iterable and has no clone() method.'.format(tensor))


timers = defaultdict(list)
cuda_timers = defaultdict(list)


class Timer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # measured in seconds
        self.interval *= 1000.0  # convert to milliseconds
        timers[self.timer_name].append(self.interval)


class CudaTimer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        cuda_timers[self.timer_name].append(self.start.elapsed_time(self.end))


def robust_min(img, p=5):
    return np.percentile(img.ravel(), p)


def robust_max(img, p=95):
    return np.percentile(img.ravel(), p)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def optimal_crop_size(max_size, max_subsample_factor, safety_margin=0):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    crop_size += safety_margin * pow(2, max_subsample_factor)
    return crop_size


def format_power(size):
    power = 1e3
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]


def flow2bgr_np(disp_x, disp_y, max_magnitude=None):
    """
    Convert an optic flow tensor to an RGB color map for visualization
    Code adapted from: https://github.com/ClementPinard/FlowNetPytorch/blob/master/main.py#L339

    :param disp_x: a [H x W] NumPy array containing the X displacement
    :param disp_y: a [H x W] NumPy array containing the Y displacement
    :returns bgr: a [H x W x 3] NumPy array containing a color-coded representation of the flow [0, 255]
    """
    assert(disp_x.shape == disp_y.shape)
    H, W = disp_x.shape

    # X, Y = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))

    # flow_x = (X - disp_x) * float(W) / 2
    # flow_y = (Y - disp_y) * float(H) / 2
    # magnitude, angle = cv.cartToPolar(flow_x, flow_y)
    # magnitude, angle = cv.cartToPolar(disp_x, disp_y)

    # follow alex zhu color convention https://github.com/daniilidis-group/EV-FlowNet

    flows = np.stack((disp_x, disp_y), axis=2)
    magnitude = np.linalg.norm(flows, axis=2)

    angle = np.arctan2(disp_y, disp_x)
    angle += np.pi
    angle *= 180. / np.pi / 2.
    angle = angle.astype(np.uint8)

    if max_magnitude is None:
        v = np.zeros(magnitude.shape, dtype=np.uint8)
        cv.normalize(src=magnitude, dst=v, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    else:
        v = np.clip(255.0 * magnitude / max_magnitude, 0, 255)
        v = v.astype(np.uint8)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = angle
    hsv[..., 2] = v
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return bgr


def get_height_width(data_loader):
    for d in data_loader:
        return d[0]['frame'].shape[-2:]  # d['events'] is a ... x H x W voxel grid


def torch2cv2(image):
    """convert torch tensor to format compatible with cv2.imwrite"""
    image = torch.squeeze(image)  # H x W
    image = image.cpu().numpy() 
    image = np.clip(image, 0, 1)
    # image = (image - np.min(image))/(np.max(image) - np.min(image))
    return (image * 255).astype(np.uint8)


def append_timestamp(path, description, timestamp):
    with open(path, 'a') as f:
        f.write('{} {:.15f}\n'.format(description, timestamp))


def setup_output_folder(output_folder):
    """
    Ensure existence of output_folder and overwrite output_folder/timestamps.txt file.
    Returns path to output_folder/timestamps.txt
    """
    ensure_dir(output_folder)
    print('Saving to: {}'.format(output_folder))
    timestamps_path = join(output_folder, 'timestamps.txt')
    open(timestamps_path, 'w').close()  # overwrite with emptiness
    return timestamps_path

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


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


def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """
    DeviceTimer = CudaTimer if device.type == 'cuda' else Timer

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():

        events_torch = torch.from_numpy(events)
        with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(device)

        with DeviceTimer('Voxel grid voting'):
            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices]
                                  * width + tis_long[valid_indices] * width * height,
                                  source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices] * width
                                  + (tis_long[valid_indices] + 1) * width * height,
                                  source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid

