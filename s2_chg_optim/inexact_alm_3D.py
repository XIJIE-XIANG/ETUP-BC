import numpy as np
from .prox_tv import prox_tv3d

def inexact_alm_3D_ON_OFF(L, inter_on, inter_off, lamb=None, tol=None, maxIter=None):

    t, m, n = inter_on.shape

    if lamb == None:
        lamb1 = 2
        lamb2 = 1 / np.sqrt(t)

    if tol == None:
        tol = 1e-5

    if maxIter == None:
       maxIter = 1000

    # --- initialize ---
    max_dual_norm = 0
    max_norm_two = 0
    for i in range(t):
        Xi = np.squeeze(inter_on[i, :, :])
        norm_two = np.linalg.svd(Xi, full_matrices=True, compute_uv=False)[0]
        norm_inf = np.linalg.norm(Xi, ord=np.inf) / lamb1
        dual_norm = max(norm_two, norm_inf.astype(np.float32))
        if dual_norm > max_dual_norm:
            max_dual_norm = dual_norm
        if norm_two > max_norm_two:
            max_norm_two = norm_two

    X = inter_on / max_dual_norm
    Y = inter_on / max_dual_norm
    Z = inter_on / max_dual_norm
    U = inter_on / max_dual_norm
    V = inter_on / max_dual_norm

    S_ON = np.zeros((t, m, n))
    S_OFF = np.zeros((t, m, n))
    M_ON = np.zeros((t, m, n))
    M_OFF = np.zeros((t, m, n))
    M_ON_ = np.zeros((t, m, n))
    M_OFF_ = np.zeros((t, m, n))

    u = 1.25 / max_norm_two
    bar = u * 1e7
    rho = 1.5

    d_norm = np.linalg.norm(np.reshape(inter_on, (t, m * n)), 'fro')

    iter = 0
    total_svd = 0
    converged = False

    sum_N_TH_ON = np.mean(inter_on, axis=0)
    sum_N_TH_OFF = np.mean(inter_off, axis=0)
    sum_N_TH = sum_N_TH_ON + sum_N_TH_OFF
    while not converged:
        iter = iter + 1

        S_ON = prox_tv3d(inter_on + M_ON - X / u, 1 / u)[0]
        S_OFF = prox_tv3d(inter_off + M_OFF - Y / u, 1 / u)[0]
        temp_S = S_ON - lamb1 / (1 * u)
        S_ON = np.where(temp_S > 0, temp_S, 0)
        temp_S = S_OFF + lamb1 / (1 * u)
        S_OFF = np.where(temp_S < 0, temp_S, 0)

        temp = (S_ON - inter_on + M_ON_) / 2 + (X + U) / (2 * u)
        temp_N = temp - lamb1 / (2 * u)
        temp_N_ = temp + lamb1 / (2 * u)
        M_ON = np.where(temp_N > 0, temp_N, 0)
        M_ON += np.where(temp_N_ < 0, temp_N_, 0)

        temp = (S_OFF - inter_off + M_OFF_) / 2 + (Y + V) / (2 * u)
        temp_N = temp - lamb1 / (2 * u)
        temp_N_ = temp + lamb1 / (2 * u)
        M_OFF = np.where(temp_N > 0, temp_N, 0)
        M_OFF += np.where(temp_N_ < 0, temp_N_, 0)

        sum_M_OFF_ = np.mean(M_OFF_, axis=0)
        temp = (L - sum_N_TH - sum_M_OFF_ + M_ON) / 2 + (Z - U) / (2 * u)
        temp_N = temp - lamb2 / (2 * u)
        temp_N_ = temp + lamb2 / (2 * u)
        M_ON_ = np.where(temp_N > 0, temp_N, 0)
        M_ON_ += np.where(temp_N_ < 0, temp_N_, 0)

        sum_M_ON_ = np.mean(M_ON_, axis=0)
        temp = (L - sum_N_TH - sum_M_ON_ + M_OFF) / 2 + (Z - V) / (2 * u)
        temp_N = temp - lamb2 / (2 * u)
        temp_N_ = temp + lamb2 / (2 * u)
        M_OFF_ = np.where(temp_N > 0, temp_N, 0)
        M_OFF_ += np.where(temp_N_ < 0, temp_N_, 0)

        sum_M_ = np.mean(M_ON_ + M_OFF_, axis=0)

        total_svd += 1

        X += u * (S_ON - inter_on - M_ON)
        Y += u * (S_OFF - inter_off - M_OFF)
        Z += u * (L - sum_N_TH - sum_M_)
        U += u * (M_ON_ - M_ON)
        V += u * (M_OFF_ - M_OFF)
        u = min(u * rho, bar)

        # stop Criterion
        sumS = np.mean(S_ON + S_OFF, axis=0)
        stopCriterion = np.linalg.norm(sumS - L, 'fro') / d_norm

        if stopCriterion < tol:
            converged = True

        # if np.mod(total_svd, 10) == 0:
            # print('|S|_TV', len(np.where(sumS != 0)[0]), '|M|_0', len(np.where(sum_M_ != 0)[0]), 'stopCriterion', stopCriterion) #, stopCriterion1, stopCriterion2, stopCriterion3)

        if not converged and iter >= maxIter:
        #     print('Maximum iterations reached')
            converged = 1

    return S_ON, S_OFF, M_ON, M_OFF

def TV_3D(D, lamb=None, tol=None, maxIter=None):
    m, n, t = D.shape

    if lamb == None:
        lamb = 1 / np.sqrt(m)

    if tol == None:
        tol = 1e-5

    if maxIter == None:
        maxIter = 10

    # --- initialize ---
    # Y = np.zeros_like(D)
    Y = D
    max_dual_norm = 0
    max_norm_two = 0
    for i in range(m):
        Y1 = np.squeeze(Y[0, :, :])
        norm_two = np.linalg.svd(Y1, full_matrices=True, compute_uv=False)[0]
        norm_inf = np.linalg.norm(Y1, ord=np.inf) / lamb
        dual_norm = max(norm_two, norm_inf.astype(np.float32))
        if dual_norm > max_dual_norm:
            max_dual_norm = dual_norm
        if norm_two > max_norm_two:
            max_norm_two = norm_two
    Y = Y / max_dual_norm

    A_hat = np.zeros((m, n, t))
    E_hat = np.zeros((m, n, t))

    mu = 1.25 / max_norm_two
    mu_bar = mu * 1e7
    rho = 1.5

    d_norm = np.linalg.norm(np.reshape(D, (m, n * t)), 'fro')

    iter = 0
    total_svd = 0
    converged = False
    stopCriterion = 1
    sv = 10
    while not converged:
        iter = iter + 1

        A_hat = prox_tv3d(D - E_hat + (1 / mu) * Y, 1 / mu)[0]

        temp_T = D - A_hat + (1 / mu) * Y
        temp_E = temp_T - lamb / mu
        temp_E_ = temp_T + lamb / mu
        E_hat = np.where(temp_E > 0, temp_E, 0)
        E_hat += np.where(temp_E_ < 0, temp_E_, 0)

        total_svd += 1
        Z = D - A_hat - E_hat
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        # stop Criterion
        stopCriterion = np.linalg.norm(np.reshape(Z, (m, n * t)), 'fro') / d_norm  # reshape(D, m, n * t)
        if stopCriterion < tol:
            converged = True

        # if np.mod(total_svd, 10) == 0:
        #     print('|A|_TV', len(np.where(A_hat != 0)), '|E|_0', len(np.where(E_hat != 0)), 'stopCriterion', stopCriterion)
        #
        # if not converged and iter >= maxIter:
        #     print('Maximum iterations reached')
        #     converged = 1

    return A_hat, E_hat