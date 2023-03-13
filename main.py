import os
import torch
import argparse
import numpy as np

from data_process.event_process import read_DV, save_events, cal_theta, divide_events_duration_count
from s1_events2chg.events2chg import E2BC, events2chg, inter_events_chg
from s2_chg_optim.grad_descent import grad_descent_ON_OFF
from s2_chg_optim.inexact_alm_3D import inexact_alm_3D_ON_OFF
from s3_gen_events.gen_events import gen_events_batch

def events_temporal_up(events_set, args):
    '''

    :param events_set: [batch_size, events_num, 4]
    :return: brightness change
             intersection of events and brightness change
    '''

    # --- load model ---
    model, crop = E2BC(args.model_path)
    model.reset_states()

    image_pre = None
    up = []

    for idx, events in enumerate(events_set):
        # --- Step1: events to brightness change ---
        chg_image, image_pre = events2chg(events, image_pre, model, crop, rangeX=args.resolution[0], rangeY=args.resolution[1])

        if idx >= 10:
            inter_on, inter_off, times, E_on, E_off, noise_on, noise_off = inter_events_chg(events, chg_image, divide=args.up_factor)

            # --- Step2: H-FPS brightness changes optimization ---
            if args.optim == 'grad_descent':
                S_on, S_off, M_on, M_off = grad_descent_ON_OFF(chg_image, inter_on, inter_off)
            elif args.optim == 'inexact_alm_3D':
                S_on, S_off, M_on, M_off = inexact_alm_3D_ON_OFF(chg_image, inter_on, inter_off)

            theta_on, theta_off = cal_theta(S_on, S_off, E_on, E_off)

            # --- Step3: Generate up-sampling events according to brightness changes ---
            cur_up = gen_events_batch(S_on, S_off, theta_on * 1, theta_off * 1, times, MAX_NUM=args.max_num, rangeX=args.resolution[0], rangeY=args.resolution[1])
            up.extend(cur_up)

            cur_noise = gen_events_batch(noise_on, noise_off, theta_on * 1.5, theta_off * 1.5, times, MAX_NUM=1, rangeX=args.resolution[0], rangeY=args.resolution[1])
            up.extend(cur_noise)

    return np.array(up)

if __name__ == '__main__':

    # --- Parameters setting ---
    parser = argparse.ArgumentParser(description='Temporal Up-sampling Events from H-FPS Brightness Change Estimation')
    parser.add_argument('-m', '--model_path', default='model/E2BC.pth', type=str, help='path to pretrained model')
    parser.add_argument('-i', '--input_file', default='data/Doraemon.aedat4', type=str, help='path to input events file')
    parser.add_argument('-o', '--output_path', default='results/', type=str, help='path to input output files')
    parser.add_argument('--max_num', default=0, type=int, help='max number of up-sampling number, default 0: the number is decided by brightness change')
    parser.add_argument('--up_factor', default=5, type=int, help='up factor of H-FPS brightness change interpolation')
    parser.add_argument('--resolution', default=[346, 260], type=list, help='spatial resolution of input events')
    parser.add_argument('--optim', default='inexact_alm_3D', type=str, help='H-FPS brightness change optimization, options: inexact_alm_3D or grad_descent')

    args = parser.parse_args()

    # --- read events ---
    events_set = read_DV(args.input_file)

    # --- output path ---
    if not os.path.exists(args.output_path):
        os.makedirs(os.path.join(args.output_path, '2D'))
        os.makedirs(os.path.join(args.output_path, '3D'))

    # --- temporal up-sampling events ---
    up = events_temporal_up(events_set, args)

    # --- save & show up-sampling results ---
    np.save(os.path.join(args.output_path, 'up.npy'), up)

    div_up = divide_events_duration_count(up, type='duration', on_off=False, duration=0.001)
    events = []
    [events.extend(events_set[i]) for i in range(len(events_set))]
    events = np.array(events)

    cnt_ori = 0
    cnt_up = 0

    for i in range(len(div_up)):

        cur_events = events[np.where((div_up[i][0, 0] <= events[:, 0]) & (events[:, 0] < div_up[i][-1, 0]))[0], :]

        cnt_ori += cur_events.shape[0]
        cnt_up += div_up[i].shape[0]

        save_events(cur_events, os.path.join(args.output_path, '2D', str(i) + '_ori'), type='2D')
        save_events(div_up[i], os.path.join(args.output_path, '2D', str(i) + '_up'), type='2D')

        save_events(cur_events, os.path.join(args.output_path, '3D', str(i) + '_ori'), type='3D')
        save_events(div_up[i], os.path.join(args.output_path, '3D', str(i) + '_up'), type='3D')

    print("Original events number: ", cnt_ori)
    print("Up-sampled events number: ", cnt_up)