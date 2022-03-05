import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def estimate_jac(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Function for estimating locally linear maps from data matrix: x_tpl = J_t @ x_t + c_t

    Args:
        X (np.ndarray): Data matrix. Assumed to be batch x time x features

    Returns:
        Tuple[np.ndarray, np.ndarray]: jacobian coefficient matrices (J_t), intercepts (c_t)
    """

    B, T, N = X.shape

    # pre-allocate memory for storing Jacobians and intercepts
    js = np.zeros(shape=(T - 1, N, N))
    cs = np.zeros(shape=(T - 1, N))

    for t in range(T - 1):

        # get neural state vectors at time t and t + 1
        X_t = X[:, t]
        X_tp1 = X[:, t + 1]

        # find the affine map that takes X_t to X_tp1VAR_results[area]
        # extract J_t and c_t from the affine map
        js[t] = reg.coef_
        cs[t] = reg.intercept_

    return js, cs


def estimate_stability_using_particle(
    js: np.ndarray, p: int, test_eigenvectors=False
) -> np.ndarray:

    """Estimate maximal lyapunov exponent given a sequence of Jacobians using the technique of ___.
    Push a random unit vector through the sequence and measure the deformation."

    Args:
        js (np.ndarray): Sequence of Jacobians, stored in a multi-dimensional array.
        p (int): Number of random unit vectors to  use.

    Returns:
        lams: p-dimensional array containing estimates of maximal Lyapunov exponent.
    """

    K, N = js.shape[0], js.shape[1]

    # generate p vectors on the unit sphere in R^n
    U = np.random.randn(N, p)
    U /= np.linalg.norm(U, axis=0)

    if test_eigenvectors:
        # generate p vectors along the eigenvectors of js[0]
        eig_vals, eig_vecs = np.linalg.eig(js[0])
        ind_max_eig = np.argmax(np.abs(eig_vals))
        leading_eig_vec = np.real(eig_vecs[:, 0])
        random_scalings = np.random.normal(0, 1, p)
        U = leading_eig_vec[:, None] * random_scalings + np.random.normal(
            0, 0.001, (N, p)
        )
        U /= np.linalg.norm(U, axis=0)

    # preallocate memory for lyapunov exponents
    lams = np.zeros(p)

    for k in tqdm(range(K)):

        # push U through jacobian at time t
        U = js[k] @ U

        # measure deformation and store log
        lams += np.log(np.linalg.norm(U, axis=0))

        # renormalize U
        U /= np.linalg.norm(U, axis=0)

    # average by number time steps to get lyapunov exponent estimates
    lams /= K

    return lams


def estimate_stability_using_particle_from_true_jac(
    W: np.ndarray, p: int, T: int, test_eigenvectors=False, gen_jac=False
) -> np.ndarray:

    """Estimate maximal lyapunov exponent given a sequence of Jacobians using the technique of ___.
    Push a random unit vector through the sequence and measure the deformation."

    Args:
        W (np.ndarray): Sequence of Jacobians, stored in a multi-dimensional array.
        p (int): Number of random unit vectors to  use.

    Returns:
        lams: p-dimensional array containing estimates of maximal Lyapunov exponent.
    """

    N = W.shape[1]

    # generate p vectors on the unit sphere in R^n
    U = np.random.randn(N, p)
    U /= np.linalg.norm(U, axis=0)

    if test_eigenvectors:
        # generate p vectors along the eigenvectors of js[0]
        eig_vals, eig_vecs = np.linalg.eig(W)
        ind_max_eig = np.argmax(np.abs(eig_vals))
        leading_eig_vec = np.real(eig_vecs[:, 0])
        random_scalings = np.random.normal(0, 1, p)
        U = leading_eig_vec[:, None] * random_scalings + np.random.normal(0, 1, (N, p))
        U /= np.linalg.norm(U, axis=0)

    # preallocate memory for lyapunov exponents
    lams = np.zeros(p)

    # choose if generalized Jacobian or identity metric
    if gen_jac:
        J = np.linalg.inv(eig_vecs) @ W @ eig_vecs
    else:
        J = W

    for t in range(T):

        # push U through jacobian at time t
        U = J @ U

        # measure deformation and store log
        lams += np.log(np.linalg.norm(U, axis=0))

        # renormalize U
        U /= np.linalg.norm(U, axis=0)

    # average by number time steps to get lyapunov exponent estimates
    lams /= T

    return lams
