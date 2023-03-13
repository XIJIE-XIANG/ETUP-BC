import numpy as np

def gen_events_batch(S_on, S_off, theta_on, theta_off, times, MAX_NUM, use_flow=False, flow=None, rangeX=346, rangeY=260):
    up = []
    for i in range(S_on.shape[0]):
        cur_up_on = gen_events(S_on[i, ...], theta_on[i], times[i][0], times[i][1] - times[i][0], MAX_NUM, use_flow=use_flow, flow=flow, rangeX=rangeX, rangeY=rangeY)
        cur_up_off = gen_events(S_off[i, ...], theta_off[i], times[i][0], times[i][1] - times[i][0], MAX_NUM, use_flow=use_flow, flow=flow, rangeX=rangeX, rangeY=rangeY)

        up.extend(cur_up_on)
        up.extend(cur_up_off)

    up = np.array(up)
    up = up[np.argsort(up[:, 0])]
    return up

def gen_events(chg, theta, t0, duration, MAX_NUM, use_flow=False, flow=None, rangeX=346, rangeY=260):

    pos = np.abs(np.where(np.round(np.abs(chg) / theta) > 1, np.round(chg / theta), 0).astype(np.int32))
    up_y, up_x = np.where(pos != 0)
    up_p = np.where(chg[up_y, up_x] > 0, 1, -1)
    up_events = []
    if use_flow == True:
        for ii in range(len(up_x)):
            if MAX_NUM != 0:
                for jj in range(min(MAX_NUM, pos[up_y[ii], up_x[ii]])):
                    cur_t = t0 + duration * np.random.random(1)[0]
                    cur_x = round(up_x[ii] + flow[1] * (cur_t - t0))
                    cur_y = round(up_y[ii] + flow[0] * (cur_t - t0))
                    if cur_x >= 0 and cur_x < rangeX and cur_y >= 0 and cur_y < rangeY:
                        up_events.append([cur_t, cur_x, cur_y, up_p[ii]])
            else:
                for jj in range(pos[up_y[ii], up_x[ii]]):
                    cur_t = t0 + duration * np.random.random(1)[0]
                    cur_x = round(up_x[ii] + flow[1] * (cur_t - t0))
                    cur_y = round(up_y[ii] + flow[0] * (cur_t - t0))
                    if cur_x >= 0 and cur_x < rangeX and cur_y >= 0 and cur_y < rangeY:
                        up_events.append([cur_t, cur_x, cur_y, up_p[ii]])
    else:
        if MAX_NUM != 0:
            [[up_events.append([t0 + duration * np.random.random(1)[0], up_x[ii], up_y[ii], up_p[ii]])
            for jj in range(min(MAX_NUM, pos[up_y[ii], up_x[ii]]))] for ii in range(len(up_x))]
        else:
            [[up_events.append([t0 + duration * np.random.random(1)[0], up_x[ii], up_y[ii], up_p[ii]])
              for jj in range(pos[up_y[ii], up_x[ii]])] for ii in range(len(up_x))]

    return np.array(up_events)
