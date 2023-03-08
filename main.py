"""
Description: Temporal up-sampling events by estimating high frame rate brightness changes
Author  : Xijie Xiang
Time    : 2023/2/17
"""

import os
import torch
import numpy as np
from tqdm import tqdm

from data_process.event_process import read_DV, save_events
from s1_events2chg.events2chg import load_model, events2chg, inter_events_chg
from s1_events2chg.util import CropParameters
from s2_chg_optim.grad_descent import grad_descent_ON_OFF
from s3_gen_events.gen_events import gen_events_batch

def events_temporal_up(events_set, divide, optim_iter, MAX_NUM, rangeX=346, rangeY=260, show=False):
    '''

    :param events_set: [batch_size, events_num, 4]
    :return: brightness change
             intersection of events and brightness change
    '''

    # --- load model ---
    model_path = '/home/xxj/projects/up-sampling/light_intensity_change/T_up_BC/model/model_best.pth'
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
    model.reset_states()

    image_pre = None

    up = []

    for idx, events in enumerate(events_set):
        # --- Step1: events to brightness change ---
        chg_image, image_pre = events2chg(events, image_pre, model, crop, rangeX=rangeX, rangeY=rangeY)

        if idx > 30: break
        if idx >= 10:
            inter_on, inter_off, theta_on, theta_off, times = inter_events_chg(events, chg_image, divide=divide)

            # --- Step2: Brightness changes optimization ---
            S_on, S_off, M_on, M_off = grad_descent_ON_OFF(chg_image, inter_on, inter_off, optim_iter=optim_iter)

            # --- Step3: Generate up-sampling events according to brightness changes ---
            cur_up = gen_events_batch(S_on, S_off, theta_on, theta_off, times, MAX_NUM, rangeX=rangeX, rangeY=rangeY)
            up.append(cur_up)

            # --- show middle results ---
            if show:
                save_events(events, 'ori_2D', type='2D')
                save_events(cur_up, 'up_2D', type='2D')

                save_events(events, 'ori_3D', type='3D')
                save_events(cur_up, 'up_3D', type='3D')
        else:
            up.append([])

    return up

if __name__ == '__main__':

    data_path = '/home/xxj/datasets/my/T_up/data/reconstruction'
    save_path = 'results/up'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # --- Parameters ---
    divide = 5
    MAX_NUM = 0
    optim_iter = 300
    rangeX, rangeY = 346, 260

    for events_file in tqdm(os.listdir(data_path)):
        if events_file in ['building2.aedat4', 'building3.aedat4', 'building32.aedat4']: continue

        # --- read events ---
        events_path = os.path.join(data_path, events_file)
        events_set = read_DV(events_path)

        # --- temporal up-sampling events ---
        up = events_temporal_up(events_set, divide, optim_iter, MAX_NUM, rangeX=rangeX, rangeY=rangeY)
        np.save(os.path.join(save_path, events_file.split('.')[0] + '.npy'), up)