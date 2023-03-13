import torch
import numpy as np

from .model import E2VIDRecurrent
from .util import CudaTimer, events_to_voxel_grid_pytorch, CropParameters
from data_process.event_process import chg_mode_0, divide_events
from data_process.TV_core import TotalVariationDenoising
from s2_chg_optim.inexact_alm_3D import TV_3D

model_info = {}
device = torch.device('cuda')

def E2BC(model_path, rangeX=346, rangeY=260):

    model_params = {
        'base_num_channels': 32,
        'num_encoders': 3,
        'num_residual_blocks': 2,
        'num_output_channels': 1,
        'kernel_size': 5,
        'skip_type': "sum",
        'norm': None,
        'num_bins': 5,
        'recurrent_block_type': "convlstm",
        'use_upsample_conv': True,
        'channel_multiplier': 2,
    }

    checkpoint = torch.load(model_path)
    model = load_model(checkpoint, model_params)
    crop = CropParameters(rangeX, rangeY, model.num_encoders)

    return model, crop

def events2rec(events, model, crop, rangeX=346, rangeY=260):

    # reconstruction images
    event_tensor = events_to_voxel_grid_pytorch(events, num_bins=5, width=rangeX, height=rangeY, device=device)
    voxel = crop.pad(event_tensor)

    with CudaTimer('Inference'):
        output = model(voxel.unsqueeze(0))['image']
        image = crop.crop(output).squeeze().cpu().numpy()

    return np.clip(image, 0, 1)

def events2chg(events, image_pre, model, crop, voxel=False, rangeX=346, rangeY=260):
    '''

    :param events: [events_num, 4] or voxel: [5, rangeY, rangeX]
    :param image_pre: last reconstruction image
    :param model:
    :param crop:
    :param voxel: input events when False, else input voxel
    :param rangeX:
    :param rangeY:
    :return: brightness change image and reconstruction image
    '''
    if ~voxel:
        events = events_to_voxel_grid_pytorch(events, num_bins=5, width=rangeX, height=rangeY, device=device)

    voxel = crop.pad(events)

    with CudaTimer('Inference'):
        output = model(voxel.unsqueeze(0))['image']
        image = crop.crop(output).squeeze().cpu().numpy()

        if image_pre is None:
            return None, image

        else:
            chg_image = image - image_pre
            chg_image = chg_mode_0(np.clip(chg_image, -1, 1))

            return chg_image, image

def load_model(checkpoint, model_kwargs):
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    logger = config.get_logger('test')

    # build model architecture
    model = E2VIDRecurrent(model_kwargs)
    logger.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def inter_events_chg(events, chg_image, divide=5):
    '''
    :param events: [events_num, 4]
    :param chg_image:
    :param divide: divide events to n(=divide) segments
    :return: n intersection of n events segments and brightness change image
    '''

    # --- divide events to n(=up_factor) small duration
    events_2D_on, events_2D_off, times = divide_events(events, divide, on_off=True)

    # --- spatial-temporal smooth of events count map ---
    events_2D_on_TV, noise_on = TV_3D(events_2D_on)
    events_2D_off_TV, noise_off = TV_3D(events_2D_off)

    idx_on_t, idx_on_y, idx_on_x = np.where(events_2D_on_TV > 0.1 * np.max(events_2D_on_TV))
    idx_off_t, idx_off_y, idx_off_x = np.where(events_2D_off_TV < -0.1 * np.min(events_2D_off_TV))

    mask_on = np.zeros_like(events_2D_on)
    mask_on[idx_on_t, idx_on_y, idx_on_x] = 1
    mask_off = np.zeros_like(events_2D_off)
    mask_off[idx_off_t, idx_off_y, idx_off_x] = 1

    # --- intersection of brightness change and smooth events count map ---
    inter_on = chg_image * mask_on
    inter_on_TV = np.where(inter_on < 0, 0, inter_on)
    inter_off = chg_image * mask_off
    inter_off_TV = np.where(inter_off > 0, 0, inter_off)

    return inter_on_TV, inter_off_TV, times, events_2D_on, events_2D_off, noise_on, noise_off