import cv2
import numpy as np
from dv import AedatFile
import matplotlib.pyplot as plt


def read_DV(events_path, gt_type=None, duration=0.0, cnt=21):
    with AedatFile(events_path) as f:

        lr = np.hstack([packet for packet in f['events'].numpy()])
        events_lr = np.array([lr['timestamp'] / 1e6, lr['x'], lr['y'], lr['polarity']]).T
        if gt_type == None:
            count = 20000
            factor = events_lr.shape[0] // count
            div_events_lr = []
            for idx in range(10, min(cnt, int(factor))):
                cur_events = events_lr[idx * count: (idx + 1) * count, :]
                div_events_lr.append(cur_events)
            return div_events_lr
        else:
            # assert gt_type != None, 'Groundtruth type needs to be specified (events or APS).'
            assert duration > 0, 'Duration needs to be bigger than 0.'
            if gt_type == 'events':
                
                hr = np.hstack([packet for packet in f['events_1'].numpy()])
                events_hr = np.array([hr['timestamp']/1e6, hr['x'], hr['y'], hr['polarity']]).T

                factor = (events_lr[-1, 0] - events_lr[0, 0]) // duration
                div_events_lr = []
                div_events_hr = []
                for i in range(10, min(cnt, int(factor))):
                    idx = np.where((events_lr[0, 0] + i * duration < events_lr[:, 0]) & (events_lr[:, 0] <= events_lr[0, 0] + (i + 1) * duration))[0]
                    cur_events = events_lr[idx, :]
                    div_events_lr.append(cur_events)

                    idx_hr = np.where((cur_events[0, 0] < events_hr[:, 0]) & (events_hr[:, 0] <= cur_events[-1, 0]))[0]
                    div_events_hr.append(events_hr[idx_hr, :])

                return div_events_lr, div_events_hr
            
            elif gt_type == 'APS':
                aps = [packet for packet in f['frames']]

                factor = (events_lr[-1, 0] - events_lr[0, 0]) // duration
                div_events_lr = []
                div_aps = []

                idx_aps = 0
                while aps[idx_aps].timestamp / 1e6 < events_lr[0, 0]:
                    idx_aps += 1
                
                for i in range(int(factor)):
                    begin_time = events_lr[0, 0] + i * duration
                    end_time = events_lr[0, 0] + (i + 1) * duration
                    idx = np.where((begin_time < events_lr[:, 0]) & (events_lr[:, 0] <= end_time))[0]
                    cur_events = events_lr[idx, :]
                    div_events_lr.append(cur_events)

                    if begin_time < aps[idx_aps].timestamp / 1e6 and aps[idx_aps].timestamp / 1e6 <= end_time:
                        div_aps.append(aps[idx_aps].image)
                        idx_aps += 1
                        if idx_aps >= 50:
                            break
                    else:
                        div_aps.append([])
                return div_events_lr, div_aps


def save_events(events, save_path, type=None, norm=True, rangeY=260, rangeX=346):

    if type == '2D':
        if events.ndim == 3:
            for i in range(events.shape[0]):
                image = events2int(events[i, ...])
                if norm:
                    image = image / max(np.abs(np.min(image)), np.max(image))
                    image = image * 127 + 128
                cv2.imwrite(save_path + '_' + str(i) + '.png', image.astype(np.uint8))
        elif events.ndim == 2:
            image = events2int(events)
            if norm:
                image = image / max(np.abs(np.min(image)), np.max(image))
                image = image * 127 + 128
            cv2.imwrite(save_path + '.png', image.astype(np.uint8))

    elif type == '3D':
        if events.ndim == 3:
            for i in range(events.shape[0]):
                cur_events = events[i, ...]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                on_idx = np.where(cur_events[:, -1] == 1)
                off_idx = np.where(cur_events[:, -1] <= 0)
                ax.scatter(cur_events[on_idx, 1], cur_events[on_idx, 0], cur_events[on_idx, 2], s=1, c='r')
                ax.scatter(cur_events[off_idx, 1], cur_events[off_idx, 0], cur_events[off_idx, 2], s=1, c='b')
                ax.set_xlabel('X')
                ax.set_xlim(0, rangeX)
                ax.set_ylabel('T')
                ax.set_ylim(events[0, 0], events[-1, 0])
                ax.set_zlabel('Y')
                ax.set_zlim(0, rangeY)
                plt.savefig(save_path + '_' + str(i) + '.png')
                plt.close()
        elif events.ndim == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            on_idx = np.where(events[:, -1] == 1)
            off_idx = np.where(events[:, -1] <= 0)
            ax.scatter(events[on_idx, 1], events[on_idx, 0], events[on_idx, 2], s=1, c='r')
            ax.scatter(events[off_idx, 1], events[off_idx, 0], events[off_idx, 2], s=1, c='b')
            ax.set_xlabel('X')
            ax.set_xlim(0, rangeX)
            ax.set_ylabel('T')
            ax.set_ylim(events[0, 0], events[-1, 0])
            ax.set_zlabel('Y')
            ax.set_zlim(0, rangeY)
            plt.savefig(save_path + '.png')
            plt.close()


def events2int(events, rangeY=260, rangeX=346):
    image = np.zeros((rangeY, rangeX))
    for i in range(events.shape[0]):
        if 0 <= round(events[i, 2]) and round(events[i, 2]) < rangeY and 0 <= round(events[i, 1]) and round(
                events[i, 1]) < rangeX:
            image[round(events[i, 2]), round(events[i, 1])] += 1 if events[i, 3] > 0 else -1

    return image


def events_on_off(events):
    idx_on = np.where(events[:, -1] > 0)[0]
    idx_off = np.where(events[:, -1] <= 0)[0]
    return events[idx_on], events[idx_off]


def chg_mode_0(chg_image):
    # 调整光强变化
    # I_0 和 I_1应有大量像素值相等，体现在光强变化C = I_1 - I_0上，就是光强变化的众数应为0。
    chg_image *= 128
    mode = np.argmax(np.bincount(
        np.reshape(chg_image.astype(np.int32) + 128, chg_image.shape[0] * chg_image.shape[1]))) - 128
    chg_image -= mode
    mode = np.argmax(np.bincount(
        np.reshape(chg_image.astype(np.int32) + 128, chg_image.shape[0] * chg_image.shape[1]))) - 128
    assert mode == 0
    chg_image /= 128.0
    return chg_image


def divide_events(events, divide, on_off=True):
    div_events = []
    if on_off:
        events_2D_on = []
        events_2D_off = []

    duration = (events[-1, 0] - events[0, 0]) / divide
    times = []

    for i in range(divide):
        start_time = events[0, 0] + i * duration
        end_time = start_time + duration
        times.append([start_time, end_time])

        idx = np.where((start_time <= events[:, 0]) & (events[:, 0] < end_time))[0]
        cur_events = events[idx, :]
        div_events.append(cur_events)

        if on_off:
            events_on, events_off = events_on_off(cur_events)
            cur_events_2D_on = events2int(events_on)
            cur_events_2D_off = events2int(events_off)

            events_2D_on.append(cur_events_2D_on)
            events_2D_off.append(cur_events_2D_off)

    if on_off:
        events_2D_on = np.stack(events_2D_on)
        events_2D_off = np.stack(events_2D_off)
        return events_2D_on, events_2D_off, times
    else:
        return div_events, times


def divide_events_duration_count(events, type, on_off=True, duration=0.001, count=15000):
    div_events = []
    if on_off:
        events_2D_on = []
        events_2D_off = []
    if type == 'duration':
        factor = (events[-1, 0] - events[0, 0]) / duration
        for idx_t in range(round(factor)):
            start_time = events[0, 0] + idx_t * duration
            end_time = start_time + duration

            idx = np.where((start_time <= events[:, 0]) & (events[:, 0] <= end_time))
            cur_events = events[idx[0], :]
            div_events.append(cur_events)

            if on_off:
                events_on, events_off = events_on_off(cur_events)
                cur_events_2D_on = events2int(events_on)
                cur_events_2D_off = events2int(events_off)

                events_2D_on.append(cur_events_2D_on)
                events_2D_off.append(cur_events_2D_off)

        if on_off:
            events_2D_on = np.stack(events_2D_on)
            events_2D_off = np.stack(events_2D_off)
            return div_events, events_2D_on, events_2D_off
        else:
            return div_events

    elif type == 'count':
        factor = events.shape[0] / count
        for idx in range(factor):
            cur_events = events[idx * count : (idx + 1) * count, :]
            div_events.append(cur_events)

            if on_off:
                events_on, events_off = events_on_off(cur_events)
                cur_events_2D_on = events2int(events_on)
                cur_events_2D_off = events2int(events_off)

                events_2D_on.append(cur_events_2D_on)
                events_2D_off.append(cur_events_2D_off)

        if on_off:
            events_2D_on = np.stack(events_2D_on)
            events_2D_off = np.stack(events_2D_off)
            return div_events, events_2D_on, events_2D_off
        else:
            return div_events


def downsample_events(events, down_factor, type='temporal'):
    if type == 'temporal':
        if isinstance(events, list):
            down_events = [events[i][::down_factor, :] for i in range(len(events))]
        elif isinstance(events, np.ndarray):
            down_events = events[::down_factor, :].tolist()

    elif type == 'spatial':
        if isinstance(events, list):
            idx = [np.where((events[i][:, 1] % 2 == 0) & (events[i][:, 2] % 2 == 0))[0] for i in range(len(events))]
            down_events = [events[i][idx[i], :] for i in range(len(events))]
        elif isinstance(events, np.ndarray):
            idx = np.where((events[:, 1] % 2 == 0) & (events[:, 2] % 2 == 0))[0]
            down_events = events[idx, :].tolist()

    return down_events


def cal_theta(S_on, S_off, E_on, E_off):

    theta_on = np.zeros(S_on.shape[0])
    theta_off = np.zeros(S_off.shape[0])

    for i in range(E_on.shape[0]):

        cur_E_on = E_on[i, ...]
        cur_E_off = E_off[i, ...]

        np.seterr(divide='ignore', invalid='ignore')
        cur_theta_on = np.where((cur_E_on != 0) & (S_on[i, ...] / cur_E_on > 0), S_on[i, ...] / cur_E_on, 0)
        theta_on[i] = np.mean(cur_theta_on[np.where(cur_theta_on > 0)])

        cur_theta_off = np.where((cur_E_off != 0) & (S_off / cur_E_off > 0), S_off / cur_E_off, 0)
        theta_off[i] = np.mean(cur_theta_off[np.where(cur_theta_off > 0)])

    return theta_on, theta_off


if __name__ == '__main__':

    # --- Test read_DV ---
    events_path = '/home/xxj/datasets/my/T_up/data/reconstruction/meeting_room.aedat4'
    div_events_lr = read_DV(events_path)
    # div_events_lr, div_events_hr = read_DV(events_path, gt=True, gt_type='events', duration=0.001)
    # div_events_lr, div_aps = read_DV(events_path, gt=True, gt_type='APS', duration=0.001)

    # --- Test save events to 2D: intensity image or 3D: events ---
    save_events(div_events_lr[0], save_path='lr_2D', type='2D')
    save_events(div_events_lr[0], save_path='lr_3D', type='3D')

    print()