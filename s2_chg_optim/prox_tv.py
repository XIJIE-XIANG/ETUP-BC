import os
import numpy as np
from datetime import datetime

def prox_tv2d(b, gamma, param=None):
    t1 = datetime.now()

    # for the GPU
    # global GLOBAL_useGPU
    #
    # if ~size(GLOBAL_useGPU, 1):
    #     GLOBAL_useGPU = 0

    # Optional input arguments
    if param == None:
        param = {}

    if not 'tol' in param:
        param['tol'] = 1e-4
    if not 'verbose' in param:
        param['verbose'] = 1
    if not 'maxit' in param:
        param['maxit'] = 2
    if not 'useGPU' in param:
        param['useGPU'] = 0
    if not 'weights' in param:
        param['weights'] = [1, 1]

    # Test of gamma
    info = {}
    if test_gamma(gamma):
        sol = b
        info['algo'] = os.getcwd()
        info['iter'] = 0
        info['final_eval'] = 0
        info['crit'] = '--'
        info['time'] = datetime.now() - t1
        return sol, info

    # if param.useGPU
    #     % gpuDevice(1);
    #     gamma = gpuArray(gamma);
    #     if isa(b, 'gpuArray')
    #         allGPU = 1;
    #     else
    #         b = gpuArray(b);
    #         allGPU = 0;
    #     end
    #     % Initializations
    #     [r, s] = gradient_op(b * 0);
    #     pold = r;
    #     qold = s;
    #     told = gpuArray(1);
    #     prev_obj = gpuArray(0);
    #     verbose = gpuArray(param.verbose);
    #     tol = gpuArray(param.tol);
    # else
    # Initializations
    [r, s] = gradient_op(b * 0)
    pold = r
    qold = s
    told = 1
    prev_obj = 0
    verbose = param['verbose']
    tol = param['tol']

    wx = param['weights'][0]
    wy = param['weights'][1]
    mt = np.max(param['weights'])

    # Main iterations
    if verbose > 1:
        print('  Proximal TV operator:\n')

    for iter in range(param['maxit']):

        # Current solution
        sol = b - gamma * div_op(r, s, wx, wy)

        # Objective function value
        tmp = gamma * np.sum(norm_tv(sol, wx, wy))
        obj = 0.5 * np.linalg.norm(b - sol, 2) ** 2 + tmp
        rel_obj = abs(obj - prev_obj) / obj
        prev_obj = obj

        # Stopping criterion
        if verbose > 1:
            print('   Iter %i, obj = %e, rel_obj = %e\n', iter, obj, rel_obj)

        if rel_obj < tol:
            crit = 'TOL_EPS'
            break

        # Udpate divergence vectors and project
        [dx, dy] = gradient_op(sol, wx, wy)

        r = r - 1 / (8 * gamma) / mt ** 2 * dx
        s = s - 1 / (8 * gamma) / mt ** 2 * dy

        weights = np.sqrt(r ** 2 + s ** 2)
        weights = np.where(weights > 1, weights, 1)

        p = r / weights
        q = s / weights

        # FISTA update
        t = (1 + np.sqrt(4 * told ** 2)) / 2
        r = p + (told - 1) / t * (p - pold)
        pold = p
        s = q + (told - 1) / t * (q - qold)
        qold = q
        told = t

    # Log after the minimization
    if not 'crit' in locals().keys():
        crit = 'MAX_IT'

    # if verbose >= 1:
    #     if param['useGPU']:
    #         print('  GPU Prox_TV 2D: obj = ', obj, ', rel_obj = ', rel_obj, ',', crit, ', iter = ', iter + 1)
    #     else:
    #         print('  Prox_TV 2D: obj = ', obj, ', rel_obj = ', rel_obj, ',', crit, ', iter = ', iter + 1)

    # if param.useGPU:
    #     if not allGPU:
    #         sol = gather(sol)
    #     info.iter = gather(iter);
    #     info.final_eval = gather(obj);
    # else:

    info['algo'] = os.getcwd()
    info['final_eval'] = obj
    info['crit'] = crit
    info['time'] = datetime.now() - t1

    return sol, info


def gradient_op(I, wx=None, wy=None):
    dx = np.concatenate([I[1:, :] - I[:-1, :], np.zeros((1, I.shape[1]))], axis=0)
    dy = np.concatenate([I[:, 1:] - I[:, :-1], np.zeros((I.shape[0], 1))], axis=1)

    if wx != None and wy != None:
        dx = dx * wx
        dy = dy * wy

    return dx, dy


def div_op(dx, dy, wx=None, wy=None):
    if wx != None and wy != None:
        dx = dx * wx.conjugate()
        dy = dy * wy.conjugate()

    I = np.concatenate(
        [np.expand_dims(dx[0, :], axis=0), dx[1:-1, :] - dx[:-2, :], np.expand_dims(-1 * dx[-1, :], axis=0)], axis=0)
    I += np.concatenate(
        [np.expand_dims(dy[:, 0], axis=1), dy[:, 1:-1] - dy[:, :-2], np.expand_dims(-1 * dy[:, -1], axis=1)], axis=1)

    return I


def norm_tv(u, wx, wy):
    if wx != None and wy != None:
        [dx, dy] = gradient_op(u, wx, wy)
    else:
        [dx, dy] = gradient_op(u)

    temp = np.sqrt(dx ** 2 + dy ** 2)

    # This allows to return a vector of norms
    return np.sum(temp)


def prox_tv3d(x, gamma, param=None):

    '''

    :param x:
    :param gamma:
    :param param:
    :return: sol = argmin_{z} gamma * ||x||_TV + 0.5*||x - z||_2^2
    '''

    t1 = datetime.now()

    # for the GPU
    global GLOBAL_useGPU

    # Optional input arguments
    if param == None:
        param = {}

    if not 'tol' in param:
        param['tol'] = 1e-4
    if not 'verbose' in param:
        param['verbose'] = 1
    if not 'maxit' in param:
        param['maxit'] = 2
    if not 'useGPU' in param:
        param['useGPU'] = 0
    if not 'weights' in param:
        param['weights'] = [1, 1, 1]
    if not 'parallel' in param:
        if x.ndim == 3:
            param['parallel'] = 1
        else:
            param['parallel'] = 0

    info = {}
    if param['parallel'] == 0:
        # call prox 3 d for each cube
        param['parallel'] = 1
        sol = np.zeros(x.shape)
        info['iter'] = 0
        info['time'] = 0
        info['algo'] = os.getcwd()
        info['final_eval'] = 0
        info['crit'] = 'TOL_EPS'
        # return this only if ALL subproblems finish with this criterion.
        param['verbose'] = param['verbose'] - 1
        # Handle verbosity
        for ii in range(x.shape[3]):
            sol[:,:,:, ii], infos_ii = prox_tv3d(x[:, :, :, ii], gamma, param)
            info['iter'] = info['iter'] + infos_ii['iter']
            info['time'] = info['time'] + infos_ii['time']
            info['final_eval'] = info['final_eval'] + infos_ii['final_eval']

            if infos_ii['crit'] == 'MAX_IT':
                info['crit'] = 'MAX_IT' # if ANY subproblem reaches maximum iterations, return this as criterion!
        return sol, info

    if test_gamma(gamma):
        sol = x
        info['algo'] = os.getcwd()
        info['iter'] = 0
        info['final_eval'] = 0
        info['crit'] = '--'
        info['time'] = datetime.now() - t1
        return sol, info

    wx = param['weights'][0]
    wy = param['weights'][1]
    wz = param['weights'][2]
    mt = np.max(param['weights'])

    # Initializations
    # if param.useGPU:
    #     # gpuDevice(1)
    #     gamma = gpuArray(gamma)
    #     if isinstance(x, 'gpuArray'):
    #         allGPU = 1
    #     else:
    #         x = gpuArray(x)
    #         allGPU = 0
    #     # Initializations
    #     [r, s, k] = gradient_op3d(x * 0)
    #     pold = r
    #     qold = s
    #     kold = k
    #     told = gpuArray(1)
    #     prev_obj = gpuArray(0)
    #     verbose = gpuArray(param.verbose)
    #     tol = gpuArray(param.tol)
    # else:
    [r, s, k] = gradient_op3d(x * 0)
    pold = r
    qold = s
    kold = k
    told = 1
    prev_obj = 0
    verbose = param['verbose']
    tol = param['tol']

    # Main iterations
    # if verbose > 1:
    #     if param['useGPU']:
    #         print('Proximal TV operator using TV:\n')
    #     else:
    #         print('Proximal TV operator:\n')

    for iter in range(param['maxit']):

        # Current solution
        sol = x - gamma * div_op3d(r, s, k, wx, wy, wz)

        # Objective function value
        x_ = np.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2]))
        sol_ = np.reshape(sol, (sol.shape[0]*sol.shape[1]*sol.shape[2]))
        obj = 0.5 * np.linalg.norm(x_ - sol_, 2) ** 2 + gamma * np.sum(norm_tv3d(sol, wx, wy, wz))
        rel_obj = abs(obj - prev_obj) / obj
        prev_obj = obj

        # Stopping criterion
        # if verbose > 1:
        #     print('   Iter %i, obj = %e, rel_obj = %e\n'.format(iter, obj, rel_obj))

        if rel_obj < tol:
            crit = 'TOL_EPS'
            break

        # Udpate divergence vectors and project
        # TODO: read reference for good explanation...We change lemma 4.2 to
        # be valid for 3D denoising and we should get a bound with 12 instead of 8.
        dx, dy, dz = gradient_op3d(sol, wx, wy, wz)
        r = r - 1 / (12 * gamma * mt ** 2) * dx
        s = s - 1 / (12 * gamma * mt ** 2) * dy
        k = k - 1 / (12 * gamma * mt ** 2) * dz
        # Isotropic TV
        weights = np.sqrt(r**2 + s**2 + k**2)
        weights = np.where(weights > 1, weights, 1)
        # anisotropic TV
        # weights = max(1, abs(r) + abs(s) + abs(k))
        p = r / weights
        q = s / weights
        o = k / weights

        # FISTA update
        t = (1 + np.sqrt(4 * (told ** 2))) / 2
        r = p + (told - 1) / t * (p - pold)
        pold = p
        s = q + (told - 1) / t * (q - qold)
        qold = q
        k = o + (told - 1) / t * (o - kold)
        kold = o
        told = t

    # Log after the minimization
    if not 'crit' in locals().keys():
        crit = 'MAX_IT'

    # if verbose >= 1:
    #     if param['useGPU']:
    #         print('  GPU Prox_TV 3D: obj = ', obj, ', rel_obj = ', rel_obj, ',',  crit, ', iter = ', iter + 1)
    #     else:
    #         print('  Prox_TV 3D: obj = ', obj, ', rel_obj = ', rel_obj, ',',  crit, ', iter = ', iter + 1)

    # if param['useGPU']:
    #     if not allGPU:
    #         sol = gather(sol)
    #     info['iter'] = gather(iter)
    #     info['final_eval'] = gather(obj)
    # else:
    info['iter'] = iter
    info['final_eval'] = obj

    info['algo'] = os.getcwd()
    info['iter'] = iter
    info['final_eval'] = obj
    info['crit'] = crit
    info['time'] = datetime.now() - t1

    return sol, info

def test_gamma(gamma):

    if gamma < 0:
        print('gamma can not be negativ!')

    if gamma == 0:
        stop = 1
    else:
        stop = 0
    return stop

def gradient_op3d(I, wx=None, wy=None, wz=None):
    dx = np.concatenate([I[1:, :, :] - I[:-1, :, :], np.zeros((1, I.shape[1], I.shape[2]))], axis=0)
    dy = np.concatenate([I[:, 1:, :] - I[:, :-1, :], np.zeros((I.shape[0], 1, I.shape[2]))], axis=1)
    dz = np.concatenate([I[:, :, 1:] - I[:, :, :-1], np.zeros((I.shape[0], I.shape[1], 1))], axis=2)

    if wx != None and wy != None and wz != None:
        dx = dx * wx
        dy = dy * wy
        dz = dz * wz

    return dx, dy, dz

def div_op3d(dx, dy, dz, wx=None, wy=None, wz=None):

    if wx != None and wy != None and wz != None:
        dx = dx * wx.conjugate()
        dy = dy * wy.conjugate()
        dz = dz * wz.conjugate()

    if dx.shape[0] == 2:
        I = np.concatenate([np.expand_dims(dx[0, :, :], axis=0), np.expand_dims(-1 * dx[-1, :, :], axis=0)], axis=0)
    else:
        I = np.concatenate([np.expand_dims(dx[0, :, :], axis=0), dx[1:-1, :, :] - dx[:-2, :, :], np.expand_dims(-1 * dx[-1, :, :], axis=0)], axis=0)

    I += np.concatenate([np.expand_dims(dy[:, 0, :], axis=1), dy[:, 1:-1, :] - dy[:, :-2, :], np.expand_dims(-1 * dy[:, -1, :], axis=1)], axis=1)
    I += np.concatenate([np.expand_dims(dz[:, :, 0], axis=2), dz[:, :, 1:-1] - dz[:, :, :-2], np.expand_dims(-1 * dz[:, :, -1], axis=2)], axis=2)

    return I

def norm_tv3d(u, wx, wy, wz):
    if wx != None and wy != None and wz != None:
        [dx, dy, dz] = gradient_op3d(u, wx, wy, wz)
    else:
        [dx, dy, dz] = gradient_op3d(u)

    temp = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # This allows to return a vector of norms
    return np.sum(temp)