# import nolds
from nolitsa import delay, lyapunov
import numpy as np
from scipy.integrate import odeint
from scipy.signal import argrelextrema
from scipy import spatial
from tqdm.auto import tqdm

def get_nn_indices(pts, p=2):
    kdTree = spatial.cKDTree(pts, leafsize=100)
    nn = kdTree.query(pts, k=2, p=p)
    nn_indices = nn[1][:, 1]
    min_dist = nn[0][:, 1]

    return nn_indices, min_dist


def get_metric(p, name=False):
    if p == 1:
        if name:
            return "cityblock"
        return spatial.distance.cityblock
    elif p == 2:
        if name:
            return "euclidean"
        return spatial.distance.euclidean
    elif p == np.Inf:
        if name:
            return "chebyshev"
        return spatial.distance.chebyshev
    else:
        if name:
            return "minkowski"
        return lambda u, v: spatial.distance.minkowski(u, v, p=p)


def calculate_fnn_stat(pts, nn_indices, r, p=2):
    metric = get_metric(p)

    N = pts.shape[0]
    nn_distances = []
    nn_next_step_distances = []

    for i in range(N - 1):
        if nn_indices[i] + 1 < N:
            nn_distance = metric(pts[i], pts[nn_indices[i]])
            if nn_distance == 0:
                nn_distances.append(1)
                nn_next_step_distances.append(np.Inf)
            else:
                nn_distances.append(nn_distance)
                nn_next_step_distances.append(metric(pts[i + 1], pts[nn_indices[i] + 1]))
    nn_distances, nn_next_step_distances = np.array(nn_distances), np.array(nn_next_step_distances)
    ratios = nn_next_step_distances / nn_distances
    fraction_greater = np.sum(ratios > r) / len(ratios)
    return fraction_greater, ratios


def lyapunov_analysis(signal_in, tau=None, nlags=1000, bins=64, max_m=5, p=1, r=10, theiler_window=60,
                                                                                    maxt=1000, method=None):
    # # pick tau
    # acf = stattools.acf(signal_in, nlags=500, adjusted='True', fft=False)
    # tau = np.argmax(acf < 0)

    # pick tau
    delayed_mi = np.zeros(nlags + 1)

    n = len(signal_in)
    for tau_ in range(nlags + 1):
        delayed_mi[tau_] = delay.mi(signal_in[tau_:], signal_in[:n - tau_], bins=bins)
    local_min_locs = argrelextrema(delayed_mi, np.less)[0]

    if tau is None:
        tau = local_min_locs[0]

    # pick m
    m_vals = np.arange(1, max_m + 1)
    fraction_fnn = np.zeros(m_vals.shape)

    for i, m in enumerate(m_vals):
        embedding = embed_signal(signal_in, m, tau=tau)
        nn_indices, _ = get_nn_indices(embedding, p=p)
        fraction_greater, ratios = calculate_fnn_stat(embedding, nn_indices, r, p=p)
        fraction_fnn[i] = fraction_greater

    if sum(fraction_fnn < 0.1) > 0:
        m = np.argmax(fraction_fnn < 0.1) + 1
    else:
        m = m_vals[-1]

    # embed data
    embedding = embed_signal(signal_in, m, tau=tau)

    if method == 'nolitsa':
        # use nolitsa to compute average divergence (average of log neighbor distance)
        d = lyapunov.mle_embed(signal_in, dim=[m], tau=tau, maxt=maxt, window=theiler_window)[0]
    if method == 'nolds':
        # nolds
        le, (k_vals, d, poly) = nolds.measures.lyap_r(signal_in, emb_dim=m, lag=tau, min_tsep=theiler_window,
                                                           trajectory_len=maxt, debug_data=True)
    else:
        # my code
        d = compute_average_neighbor_distance(signal_in, m, tau, max_delta=maxt, num_reference_pts=None, p=p,
                                                                           min_tsep=theiler_window, progress_bar=False)

    t = np.arange(maxt + 1)  # ms

    return dict(
        # acf=acf,
        delayed_mi=delayed_mi,
        bins=bins,
        local_min_locs=local_min_locs,
        tau=tau,
        m_vals=m_vals,
        fraction_fnn=fraction_fnn,
        m=m,
        embedding=embedding,
        d=d,
        t=t
    )


def average_neighbor_distance(embedding, pairwise_dists, max_delta=30, num_reference_pts=100, min_tsep=60,
                              progress_bar=False, iterator=None):
    S = np.zeros(max_delta + 1)
    N = len(embedding)
    if progress_bar:
        iterator = tqdm(total=np.sum((N - np.arange(max_delta + 1)) * (N - np.arange(max_delta + 1))))

    if num_reference_pts is None:
        reference_indices = np.arange(N - max_delta)
    else:
        reference_indices = np.random.choice(np.arange(N - max_delta), size=(num_reference_pts,))

    counts = np.zeros(max_delta + 1)

    for i in reference_indices:
        pairwise_dists[i, i] = np.Inf
        pairwise_dists[i, max(0, i - min_tsep):min(i + min_tsep, N)] = np.Inf
        pairwise_dists[i, -(max_delta + 1):] = np.Inf
        nb_idx = np.argmin(pairwise_dists[i, :])

        for delta_n in range(max_delta + 1):
            traj_dist = pairwise_dists[i + delta_n, nb_idx + delta_n]
            if traj_dist != 0:
                S[delta_n] += np.log(traj_dist)
                counts[delta_n] += 1

        if iterator is not None:
            iterator.update()

    S = S / counts
    if iterator is not None and progress_bar:
        iterator.close()

    return S, np.arange(max_delta + 1)


def compute_average_neighbor_distance(signal_in, m, tau, max_delta=30, num_reference_pts=100, p=1, min_tsep=60,
                                      progress_bar=True):
    results = []
    total = 0
    if progress_bar:
        embedding = embed_signal(signal_in, m, tau=tau)
        N = len(embedding)
        if num_reference_pts is None:
            total += (N - max_delta)
        else:
            total += num_reference_pts
        iterator = tqdm(total=total)

    embedding = embed_signal(signal_in, m, tau=tau)
    pairwise_dists = spatial.distance.cdist(embedding, embedding, metric=get_metric(p, name=True))
    if progress_bar:
        S, delta_n = average_neighbor_distance(embedding, pairwise_dists, max_delta=max_delta,
                                               num_reference_pts=num_reference_pts, min_tsep=min_tsep,
                                               progress_bar=False, iterator=iterator)
    else:
        S, delta_n = average_neighbor_distance(embedding, pairwise_dists, max_delta=max_delta,
                                               num_reference_pts=num_reference_pts, min_tsep=min_tsep)

    if progress_bar:
        iterator.close()

    return S


def lyap_spectrum_QR(Js, T, debug=False):
    K, n = Js.shape[0], Js.shape[-1]
    old_Q = np.eye(n)
    H = np.eye(n)

    lexp = np.zeros(n, dtype="float32")
    lexp_counts = np.zeros(lexp.shape)

    iterator = None
    if debug:
        print("Computing Lyapunov spectrum")
        iterator = tqdm(total=K)

    for t in range(K):
        H = Js[t] @ H

        # QR-decomposition of T * old_Q
        mat_Q, mat_R = np.linalg.qr(np.dot(Js[t], old_Q))
        # force diagonal of R to be positive
        # (if QR = A then also QLL'R = A with L' = L^-1)
        sign_diag = np.sign(np.diag(mat_R))
        sign_diag[np.where(sign_diag == 0)] = 1
        sign_diag = np.diag(sign_diag)
        mat_Q = np.dot(mat_Q, sign_diag)
        mat_R = np.dot(sign_diag, mat_R)

        old_Q = mat_Q

        # successively build sum for Lyapunov exponents
        diag_R = np.diag(mat_R)

        # filter zeros in mat_R (would lead to -infs)
        idx = np.where(diag_R > 0)
        lexp_i = np.zeros(diag_R.shape, dtype="float32")
        lexp_i[idx] = np.log(diag_R[idx])
        lexp_i[np.where(diag_R == 0)] = np.inf

        lexp[idx] += lexp_i[idx]
        lexp_counts[idx] += 1

        if debug:
            iterator.update()
    if debug:
        iterator.close()

    # it may happen that all R-matrices contained zeros => exponent really has
    # to be -inf

    # normalize exponents over number of individual mat_Rs
    # lexp[idx] /= lexp_counts[idx]
    lexp[np.where(lexp_counts == 0)] = np.inf

    lexp /= T

    return lexp

# =================================
# SIMULATE SPECIFIC SYSTEMS
# =================================

# T, dt are in seconds
def simulate_lorenz(T=50, dt=0.01, rho=28, beta=8 / 3, sigma=10, initial_condition=None):
    if initial_condition is None:
        initial_condition = np.random.normal(size=(3,))

    time_vals = np.arange(0, T, dt)
    pts = np.zeros((len(time_vals), 3))
    pts[0] = initial_condition

    for t in range(1, len(time_vals)):
        x, y, z = pts[t - 1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        x += dx * dt
        y += dy * dt
        z += dz * dt

        pts[t] = [x, y, z]

    return pts

# T is in seconds
def simulate_baier_klein(T=5000, initial_condition=None):
    if initial_condition is None:
        initial_condition = np.random.normal(size=(3,))
    
    time_vals = np.arange(0, T)
    pts = np.zeros((len(time_vals), 3))
    pts[0] = initial_condition
    
    for t in range(1, len(time_vals)):
        x, y, z = pts[t - 1]
        x_new = 1.76 - y**2 - 0.1*z
        y_new = x
        z_new = y
        
        pts[t] = [x_new, y_new, z_new]
    
    return pts

# T, dt are in seconds
def simulate_rossler(T=100, dt=0.02, a=0.1, b=0.1, c=14, initial_condition=None):
    if initial_condition is None:
        initial_condition = np.random.normal(size=(3,))

    time_vals = np.arange(0, T, dt)
    pts = np.zeros((len(time_vals), 3))
    pts[0] = initial_condition
    
    for t in range(1, len(time_vals)):
        x, y, z = pts[t - 1]
        dx = -y - z
        dy = x + a*y
        dz = b + z*(x - c)
        
        x += dx * dt
        y += dy * dt
        z += dz * dt

        pts[t] = [x, y, z]
    
    return pts

# T, dt are in seconds
# Pendulum rod lengths (m), bob masses (kg).
# g is the gravitational acceleration (m.s-2).
def simulate_double_pendulum(T=30, dt=0.01, L1=1, L2=1, m1=1, m2=1, g=9.81):

    def deriv(y, t, L1, L2, m1, m2):
        """Return the first derivatives of y = theta1, z1, theta2, z2."""
        theta1, z1, theta2, z2 = y

        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

        theta1dot = z1
        z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
                 (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
        theta2dot = z2
        z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
                 m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
        return theta1dot, z1dot, theta2dot, z2dot

    def calc_E(y):
        """Return the total energy of the system."""

        th1, th1d, th2, th2d = y.T
        V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
        T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
                2*L1*L2*th1d*th2d*np.cos(th1-th2))
        return T + V

    t = np.arange(0, T, dt)
    # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
    y0 = np.array([3*np.pi/7, 0, 3*np.pi/4, 0])

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

    # Check that the calculation conserves total energy to within some tolerance.
    EDRIFT = 0.05
    # Total energy from the initial conditions
    E = calc_E(y0)
    if np.max(np.sum(np.abs(calc_E(y) - E))) > EDRIFT:
        sys.exit('Maximum energy drift of {} exceeded.'.format(EDRIFT))

    # Unpack z and theta as a function of time
    theta1, theta2 = y[:,0], y[:,2]

    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    
    return np.array([x1, y1, x2, y2]).T
