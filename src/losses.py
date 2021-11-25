import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

from .utils import A_0, nm, kT, slm


def dist_loss(W, separation, target_dist, a_0=A_0 * nm):
    x = separation / nm
    global_min_kT = (W / kT).min()

    # Find all local minima in the work curve
    min_idxs = argrelextrema(W / kT, np.less)[0]

    # consider only minima a distance greater than half an optical binding distance
    mask = np.where(separation / nm > a_0 / 2 / nm)[0]
    min_idxs = np.array([idx for idx in min_idxs if idx in mask])

    # If there are no local, define the loss as the difference between the global minima of the work-curve to the work
    # at the target separation
    if min_idxs.size == 0:
        f = interp1d(x, W / kT)
        target_kT = f(target_dist / nm).item()
        min_kT = W.min() / kT
        loss = target_kT - min_kT
        return loss

    # Check if there is a minima very close to the target separation
    is_in_n_sep = np.logical_and(
        x[min_idxs] < (target_dist / nm + a_0 / nm / 16),
        x[min_idxs] > (target_dist / nm - a_0 / nm / 16),
    )

    if np.any(is_in_n_sep):
        # if there is, find the work at that minima
        min_idx = (x[min_idxs][is_in_n_sep] - target_dist / nm).argmin()
        min_kT = W[min_idxs][is_in_n_sep][min_idx] / kT

        other_minimas = np.delete(
            W[min_idxs] / kT, np.where(min_kT == (W[min_idxs] / kT))[0][0]
        )

        if other_minimas.size == 0:

            # If there are no other local minma, check if the target minima is the global minma
            if min_kT > global_min_kT:
                # ???What to do if it is the only local minima???
                # If not, make the loss the difference between the target minima and the global minima
                loss = min_kT - global_min_kT
            else:
                # ???What to do if it IS? the global minima already???

                # other_kT = min(W[0], W[-1]) / kT
                # loss = min_kT - other_kT

                # loss is difference betwen the global maxima and global minima
                max_kT = (W / kT).max()
                loss = min_kT - max_kT

                # f = interp1d(x, W / kT)
                # other_kT = f(A_1).item()
                # loss = min_kT - other_kT
        else:

            # min_2nd = np.delete(
            #    W[min_idxs] / kT, np.where(min_kT == (W[min_idxs] / kT))[0][0]
            # ).min()
            if global_min_kT < min_kT:
                # If target minima is not global minima, Loss is difference between global min and target local min

                loss = min_kT - global_min_kT
            else:
                # If target minima is already global minima, maximize separation between the target and the competitor

                loss = min_kT - other_minimas.min()

    else:
        # If there are no minima, loss is difference between global minima and work at target separation
        f = interp1d(x, W / kT)
        target_kT = f(target_dist / nm).item()
        loss = target_kT - global_min_kT

    return loss


def angle_loss_smooth(W, thetas, target_angle):
    # Angle loss is simply distance between the global min and target
    f = interp1d(thetas, W / kT)
    target_kT = f(target_angle).item()
    min_kT = (W / kT).min()
    loss = target_kT - min_kT

    return loss  # + size_loss


def phase_loss(anm, Nmax, rho_scale):
    Nx = 256
    theta = np.linspace(0, 1, Nx)
    phi = np.linspace(0, 2 * np.pi, Nx)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    # x = np.sin(THETA) * np.cos(PHI)
    # y = np.sin(THETA) * np.sin(PHI)
    phase = slm(THETA, PHI, anm, Nmax, rho_scale)
    return phase.max() - phase.min()
