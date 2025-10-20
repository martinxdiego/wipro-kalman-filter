from __future__ import annotations
import numpy as np
from .base import FilterState
from numba import njit
import numba

numba.set_num_threads(4)  # oder 8, je nach CPU

@njit
def joseph_update(P, K, H, R):
    # Joseph stabilized covariance update
    I = np.eye(P.shape[0])
    t = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    # symmetrize to curb numerical drift
    return 0.5 * (t + t.T)

class EKF:
    """EKF for (potentially) nonlinear models.
    Here we accept callables f, F (Jacobian), h, H (Jacobian).
    """
    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        self.state = FilterState(x=x0.astype(float), P=P0.astype(float))

    def predict(self, f, F, Q, u=None):
        x, P = self.state.x, self.state.P
        x_pred = f(x, u)
        Fk = F(x, u)
        P_pred = Fk @ P @ Fk.T + Q
        self.state = FilterState(x=x_pred, P=P_pred)
        return self.state

    def update(self, z, h, H, R):
        if np.isnan(z):
            return self.state  # missing measurement, skip
        x, P = self.state.x, self.state.P
        y = np.atleast_1d(z - h(x))   # Innovation als 1D-Array
        Hk = H(x)
        S = Hk @ P @ Hk.T + R
        S = 0.5 * (S + S.T)
        K = P @ Hk.T @ np.linalg.pinv(S)
        x_new = x + (K @ y).reshape(x.shape)
        P_new = joseph_update(P, K, Hk, R)
        self.state = FilterState(x=x_new, P=P_new)
        return self.state
