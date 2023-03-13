import os
import cv2
import torch
import random
import numbers
import numpy as np
from torch.utils.data import Dataset

class DatasetECOCO(Dataset):
    """Load sequences of time-synchronized {event tensors + frames} from a folder."""
    def __init__(self, dataset_patams={}):
        self.L = dataset_patams['sequence_length']
        assert (self.L > 0)
        self.data_path = dataset_patams['data_path']

        self.transform = None
        self.vox_transform = self.transform

    def __len__(self):
        return len(os.listdir(self.data_path))

    def get_item(self, seq_path, index, seed=None):
        """
        Get data at index.
            :param index: index of data
            :param seed: random seed for data augmentation
        """
        assert 0 <= index < len(os.listdir(os.path.join(seq_path, "VoxelGrid-betweenframes-5"))) - 3, "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        seed = random.randint(0, 2 ** 32) if seed is None else seed

        flow_path = os.path.join(seq_path, "flow")
        frame_path = os.path.join(seq_path, "frames")
        event_path = os.path.join(seq_path, "VoxelGrid-betweenframes-5")

        flow = np.load(os.path.join(flow_path, "disp01_{:>010d}.npy".format(index + 1)))
        event = np.load(os.path.join(event_path, "event_tensor_{:>010d}.npy".format(index)))
        frame = np.expand_dims(cv2.imread(os.path.join(frame_path, "frame_{:>010d}.png".format(index + 1)), -1), axis=0)

        frame = self.transform_frame(frame, seed)
        flow = self.transform_flow(flow, seed)
        bi_event = self.transform_voxel(event, seed)

        item = {'frame': frame,
                'flow': flow,
                'events': bi_event}

        return item

    def transform_frame(self, frame, seed):
        """
        Augment frame and turn into tensor
        """
        frame = torch.from_numpy(frame).float() / 255
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        voxel = torch.from_numpy(voxel)
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_flow(self, flow, seed):
        """
        Augment flow and turn into tensor
        """
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_flow=True)
        return flow

    def __getitem__(self, i, seed=None):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        assert (i >= 0)
        assert (i < self.__len__())

        i += 950
        sequence = []

        seq_path = os.path.join(self.data_path, "sequence_{:>010d}".format(i))

        # add the first element (i.e. do not start with a pause)
        k = 0
        j = 0
        item = self.get_item(seq_path, j)
        sequence.append(item)

        for n in range(self.L - 1):
            k += 1
            item = self.get_item(seq_path, j + k)
            sequence.append(item)

        return sequence

class Dataset6DOF(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return

    def __getitem__(self, item):
        return

class DatasetERGB(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return

    def __getitem__(self, item):
        return

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> torchvision.transforms.Compose([
        >>>     torchvision.transforms.CenterCrop(10),
        >>>     torchvision.transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, is_flow=False):
        for t in self.transforms:
            x = t(x, is_flow)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomCrop(object):
    """Crop the tensor at a random location.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    @staticmethod
    def get_params(x, output_size):
        w, h = x.shape[2], x.shape[1]
        th, tw = output_size
        if th > h or tw > w:
            raise Exception("Input size {}x{} is less than desired cropped \
                    size {}x{} - input tensor shape = {}".format(w,h,tw,th,x.shape))
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def __call__(self, x, is_flow=False):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """
        i, j, h, w = self.get_params(x, self.size)

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        return x[:, i:i + h, j:j + w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Center crop the tensor to a certain size.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    def __call__(self, x, is_flow=False):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """
        w, h = x.shape[2], x.shape[1]
        th, tw = self.size
        assert(th <= h)
        assert(tw <= w)
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve
            # the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        return x[:, i:i + th, j:j + tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)