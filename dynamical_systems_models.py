import numpy as np
import pandas as pd
import scipy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
import time
from tqdm.auto import tqdm

from nld_utils import embed_signal

# =======================================
# VAR(p)
# =======================================

def compute_VAR_p(window_data, p=1, unit_indices=None, lamb=0, PCA_dim=-1):
    if unit_indices is None:
        chunk = window_data
    else:
        chunk = window_data[:, unit_indices]

    results = {}
    results['explained_variance'] = None
    if PCA_dim > 0:
        if PCA_dim < 2:
            raise ValueError(f"PCA dimension must be greater than 1; provided value was {PCA_dim}")
        pca = PCA(n_components=PCA_dim)
        chunk = pca.fit_transform(chunk)
        results['explained_variance'] = pca.explained_variance_ratio_
    
    if lamb == 0:
        model = VAR(chunk)
        VAR_results = model.fit(p)

        results['coefs'] = VAR_results.coefs
        results['intercept'] = VAR_results.intercept
    else:
        # X_p = np.zeros((N*p + 1, window - p))
        # Y_p = np.zeros((N*p, window - p))
        # for t in range(window - p):
        #     for i in range(p):
        #         X_p[i*N:(i + 1)*N, t] = chunk[t + p - 1 - i]
        #         Y_p[i*N:(i + 1)*N, t] = chunk[t + p - i]
        # # embedded_signal = 
        # X_p[-1] = np.ones(window - p)

        window = chunk.shape[0]
        N = chunk.shape[1]
        embedded_signal = embed_signal(chunk, p, 1, direction='reverse')
        X_p = embedded_signal[:-1].T
        Y_p = embedded_signal[1:].T
        X_p = np.vstack([X_p, np.ones((1, window - p))])
        U, S, Vh = np.linalg.svd(X_p)

        S_mat_inv = np.zeros((window - p, N*p + 1))
        S_mat_inv[np.arange(len(S)), np.arange(len(S))] = S/(S**2 + lamb)
        full_mat = Y_p[:N] @ Vh.T @ S_mat_inv @ U.T
        coefs = np.zeros((p, N, N))
        for j in range(p):
            coefs[j] = full_mat[:, j*N:(j + 1)*N]
        results['coefs'] = coefs
        results['intercept'] = full_mat[:, -1]

    N = chunk.shape[1]
    A_mat = np.zeros((N*p, N*p))
    for i in range(p):
        A_mat[0:N][:, i*N:(i+1)*N] = results['coefs'][i]

    for i in range(p - 1):
        A_mat[(i + 1)*N:(i + 2)*N][:, i*N:(i + 1)*N] = np.eye(N)
    e = np.linalg.eigvals(A_mat)   
    results['eigs'] = e  
    results['criticality_inds'] = np.abs(e)

    try:
        results['info_criteria'] = VAR_results.info_criteria
    except:
        results['info_criteria'] = None

    return results

def predict_VAR_p(data, coefs, intercept, unit_indices=None, persistence_baseline=False, PCA_dim=-1):
    if unit_indices is None:
        chunk = data
    else:
        chunk = data[:, unit_indices]

    if PCA_dim > 0:
        if PCA_dim < 2:
            raise ValueError(f"PCA dimension must be greater than 1; provided value was {PCA_dim}")
        pca = PCA(n_components=PCA_dim)
        chunk = pca.fit_transform(chunk)
    
    # BUILD PARAMS FOR PREDICTION
    p = coefs.shape[0]
    n = chunk.shape[1]
    
    params = np.zeros((1 + n*p, n))
    params[0] = intercept
    for i in range(p):
        params[1 + i*n:1 + (i + 1)*n] = coefs[i].T

    # LAG DATA
    lagged_data = np.zeros((chunk.shape[0] - p, chunk.shape[1]*p + 1))
    lagged_data[:, 0] = np.ones(lagged_data.shape[0])
    for i in range(p):
        lagged_data[:, i*chunk.shape[1] + 1:(i + 1)*chunk.shape[1] + 1] = chunk[p - 1 - i:chunk.shape[0] - 1 - i]
    
    # PREDICT
    prediction = lagged_data @ params
    true_vals = chunk[p:]
    
    if not persistence_baseline:
        return prediction, true_vals
    else:
        return prediction, true_vals, ((chunk[p-1:-1] - chunk[p:])**2).mean()

# =======================================
# EDMD
# =======================================

def ravel_hermite_multi_index(multi_index, max_order, grouped_by):
    g = int(np.argmax(np.array(multi_index) != 0)/grouped_by) # active group number
    index_base = ((max_order + 1)**grouped_by)*g
    
    n = min([len(multi_index) - g*grouped_by, grouped_by])
    shape_tuple = tuple([max_order + 1]*n)
    
    index_base += np.ravel_multi_index(multi_index[g*grouped_by:g*grouped_by + n], shape_tuple) - g
    
    return index_base

def unravel_hermite_index(index, shape_tuple, max_order, grouped_by):
    # print(f"index = {index}")
    # print(f"N = {N}")

    if index == 0:
        g = 0
    else:
        g = int(np.floor((index - 1)/((max_order + 1)**grouped_by - 1)))
    
    raveled_group_index = index - ((max_order + 1)**grouped_by - 1)*g
    
    # print(f"g = {g}")
    # print(f"raveled_group_index = {raveled_group_index}")

    N = len(shape_tuple)
    multi_index = np.zeros(N, dtype=int)
    
    n = min([N - g*grouped_by, grouped_by])
    group_shape = shape_tuple[g*grouped_by:g*grouped_by + n]

    multi_index[g*grouped_by:g*grouped_by + n] = np.unravel_index(raveled_group_index, group_shape)
    
    return multi_index

def hermite_dictionary(signal, max_order=-1, grouped_by=None, include_signal=None):
    if max_order > 0:
        if grouped_by is None:
            grouped_by = signal.shape[1]
        N_groups = int(np.ceil(signal.shape[1]/grouped_by))
        N_funcs = ((max_order + 1)**grouped_by - 1)*N_groups + 1

        dictionary_components = np.zeros((max_order + 1, signal.shape[1], signal.shape[0]))
        for order in range(max_order + 1):
            for i in range(signal.shape[1]):
                dictionary_components[order, i] = scipy.special.eval_hermitenorm(order, signal[:, i])

        dictionary = np.zeros((signal.shape[0], N_funcs))

        shape_tuple = tuple([max_order + 1]*signal.shape[1])

        for d in range(N_funcs):
            dictionary[:, d] = np.ones(signal.shape[0])
            unraveled_index = unravel_hermite_index(d, shape_tuple, max_order, grouped_by)
            for i in range(signal.shape[1]):
                order = unraveled_index[i]
                dictionary[:, d] *= dictionary_components[order, i]
        
        data = dictionary
    elif max_order == 0:
        data = np.ones((signal.shape[0], 1))
    else:
        data = np.hstack([signal, np.ones((signal.shape[0], 1))])

    if include_signal is not None:
        data = np.hstack([data, include_signal])

    return data

def thin_plate_spline_dictionary(signal, n_clusters=100, include_signal=False, cluster_centers=None):
    if cluster_centers is None:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans_fit = kmeans.fit(signal)
        cluster_centers = kmeans_fit.cluster_centers_
    dist_mat = scipy.spatial.distance_matrix(signal, cluster_centers)
    dist_mat[dist_mat == 0] = 1
    dictionary = np.multiply(dist_mat**2, np.log(dist_mat))
    if include_signal:
        dictionary = np.hstack([dictionary, signal])
    
    return dictionary, cluster_centers

def construct_dictionary(signal, method='hermite', max_order=-1, grouped_by=None, n_clusters=100, include_signal=False, cluster_centers=None, normalize=False):
    # print(signal.shape)
    if method == 'hermite':
        if isinstance(include_signal, bool):
            if not include_signal:
                include_signal = None
            else:
                raise ValueError("include_signal is True but hermite_dictionary requires an actual signal input")
        data = hermite_dictionary(signal, max_order=max_order, grouped_by=grouped_by, include_signal=include_signal)
        cluster_centers = None
    else: # method == 'thin_plat_spline'
        data, cluster_centers = thin_plate_spline_dictionary(signal, n_clusters=n_clusters, include_signal=include_signal, cluster_centers=cluster_centers)  

    if normalize:
        for i in range(data.shape[1]):
            sd = data[:, i].std()
            if sd != 0:
                data[:, i] = (data[:, i] - data[:, i].mean())/sd
    return data, cluster_centers

def compute_EDMD(window_data, p=1, lamb=0, PCA_dim=-1, dictionary_method='hermite', max_order=-1, grouped_by=None, n_clusters=100, include_signal=False, normalize=True, unit_indices=None):
    if unit_indices is None:
        chunk = window_data
    else:
        chunk = window_data[:, unit_indices]

    if include_signal and dictionary_method=='hermite':
        # include_signal = chunk[p-1:].copy()
        include_signal = embed_signal(chunk, p, 1, direction='reverse')

    results = {}
    results['explained_variance'] = None
    if PCA_dim > 0:
        if PCA_dim < 2:
            raise ValueError(f"PCA dimension must be greater than 1; provided value was {PCA_dim}")
        pca = PCA(n_components=PCA_dim)
        chunk = pca.fit_transform(chunk)
        results['explained_variance'] = pca.explained_variance_ratio_
    
    window = chunk.shape[0]
    N = chunk.shape[1]

    embedded_signal = embed_signal(chunk, p, 1, direction='reverse')
    embedded_signal_d, cluster_centers = construct_dictionary(embedded_signal, method=dictionary_method, max_order=max_order, grouped_by=grouped_by, n_clusters=n_clusters, include_signal=include_signal, normalize=normalize)
    # print(embedded_signal_d.shape)
    X_d = embedded_signal_d[:-1].T
    Y_d = embedded_signal_d[1:].T
    D = X_d.shape[0]

    if D <= window:
        # STANDARD DMD METHOD WITH SVD
        U, S, Vh = np.linalg.svd(X_d)
        # set very small singular values to zero
        # S = S[S > 1e-7]
        S_mat_inv = np.zeros((window - p, D))
        S_mat_inv[np.arange(len(S)), np.arange(len(S))] = S/(S**2 + lamb)
        # return Y_d, Vh, S_mat_inv, U, S
        full_mat = Y_d @ Vh.T @ S_mat_inv @ U.T
        results['coefs'] = full_mat
        results['intercept'] = None
        results['U'] = U
        A_mat = full_mat
    else: # D > window
        # METHOD OF SNAPSHOTS (ASSUMES THAT window <<< D)
        eigvals, eigvecs = np.linalg.eig(X_d.T @ X_d)
        # eigvals of cov mat should be real and positive, errors should be small on order of 1e-10
        S = np.sqrt(np.clip(np.real(eigvals), 0, np.Inf)) 
        S = np.sqrt(np.clip(np.real(eigvals), 0, np.Inf))
        S_vals_inv = np.array([1/s if s != 0 else s for s in S])
        VS_inv = np.real(eigvecs) @ np.diag(S_vals_inv)
        U = X_d @ VS_inv
        A_tilde = U.T @ Y_d @ VS_inv
        # X_d_tilde = U.T @ X_d # project X onto the left singular vectors

        results['coefs'] = A_tilde
        results['intercept'] = None
        results['U'] = U
        A_mat = A_tilde
    
    e = np.linalg.eigvals(A_mat)   
    results['eigs'] = e  
    results['criticality_inds'] = np.abs(e)
    results['p'] = p
    results['lamb'] = lamb
    results['PCA_dim'] = PCA_dim
    results['dictionary_method'] = dictionary_method
    results['max_order'] = max_order
    results['grouped_by'] = grouped_by
    results['n_clusters'] = n_clusters
    results['include_signal'] = include_signal
    results['normalize'] = normalize
    results['cluster_centers'] = cluster_centers

    # try:
    #     results['info_criteria'] = VAR_results.info_criteria
    # except:
    #     results['info_criteria'] = None

    return results

def predict_EDMD(data, coefs, p, PCA_dim=-1, intercept=None, U=None, dictionary_method='hermite', max_order=-1, grouped_by=None, include_signal=False, cluster_centers=None, normalize=False, unit_indices=None, predict_type='all', tail_bite=False, persistence_baseline=False):
    if unit_indices is None:
        chunk = data
    else:
        chunk = data[:, unit_indices]

    if include_signal and dictionary_method=='hermite':
        # include_signal = chunk[p-1:].copy()
        include_signal = embed_signal(chunk, p, 1, direction='reverse')
        n_orig = chunk.shape[1]

    if PCA_dim > 0:
        if PCA_dim < 2:
            raise ValueError(f"PCA dimension must be greater than 1; provided value was {PCA_dim}")
        pca = PCA(n_components=PCA_dim)
        chunk = pca.fit_transform(chunk)

    # BUILD PARAMS FOR PREDICTION
    n = chunk.shape[1]
    # p = int((np.log10(coefs.shape[0])/np.log10(max_order))/n)

    params = coefs

    # LAG DATA
    # lagged_data = np.zeros((chunk.shape[0] - p, n*p))
    # for i in range(p):
    #     lagged_data[:, i*chunk.shape[1]:(i + 1)*chunk.shape[1]] = chunk[p - 1 - i:chunk.shape[0] - 1 - i]
    # lagged_data_d = construct_dictionary(lagged_data, max_order=max_order)
    embedded_signal = embed_signal(chunk, p, 1, direction='reverse')
    embedded_signal_d, cluster_centers = construct_dictionary(embedded_signal, method=dictionary_method, max_order=max_order, grouped_by=grouped_by, cluster_centers=cluster_centers, include_signal=include_signal, normalize=normalize)
    if U is not None:
        embedded_signal_d = embedded_signal_d @ U
    X_d_T = embedded_signal_d[:-1]
    Y_d_T = embedded_signal_d[1:]

    actual_indices = None
    if predict_type == 'actual':
        if U is not None:
            raise ValueError("U is not None. To do actual prediction the dynamic matrix A cannot be on a low dimensional projected space.")
        if dictionary_method == 'hermite':
            if isinstance(include_signal, bool):
                if not include_signal:
                    shape_tuple = ()
                    for i in range(n*p):
                        shape_tuple += (max_order + 1,)
                    actual_indices = np.zeros(n, dtype=int)
                    for i in range(n):
                        actual_multi_ind = np.zeros(n*p, dtype=int)
                        actual_multi_ind[i] = 1
                        actual_indices[i] = ravel_hermite_multi_index(actual_multi_ind, max_order, grouped_by)
            else:
                actual_indices = np.arange(-n_orig*p, -n_orig*p + n_orig, 1)
        else: # dictionary_method == 'thin_plate_spline'
            if not include_signal:
                raise ValueError("Cannot do actual prediction because signal is not included.")
            actual_indices = np.arange(-n*p, -n*p + n, 1)

    # PREDICT
    if not tail_bite:
        # prediction_d = X_d_T @ params
        prediction_d = (params @ X_d_T.T).T
    else:
        prediction_d = np.zeros(X_d_T.shape)
        for t in range(prediction_d.shape[0]):
            if t == 0:
                prediction_d[t] = (params @ X_d_T[0].T).T
            else:
                prediction_d[t] = (params @ prediction_d[t - 1].T).T
        
    if predict_type == 'all':
        prediction = prediction_d
        true_vals = Y_d_T
        pb = X_d_T
        pb_mse = ((X_d_T - Y_d_T)**2).mean()
    else: # predict_type == 'actual'
        prediction = prediction_d[:, actual_indices]
        true_vals = Y_d_T[:, actual_indices]
        # pb_mse = ((chunk[p-1:-1] - chunk[p:])**2).mean()
        pb = X_d_T[:, actual_indices]
        pb_mse = ((X_d_T[:, actual_indices] - true_vals)**2).mean()
    if not persistence_baseline:
        return prediction, true_vals
    else:
        return prediction, true_vals, pb_mse, pb

# =======================================
# Large Scale Analysis Functions
# =======================================

def get_method_funcs(method, lamb, dictionary_method, max_order, grouped_by, n_clusters, include_signal, normalize):
    if method == 'VAR(p)':
        compute_func = lambda sig, lag: compute_VAR_p(sig, lag, lamb=lamb)
        predict_func = lambda sig, coefs, p, intercept: predict_VAR_p(sig, coefs, intercept, persistence_baseline=True)
    elif method == 'EDMD':
        compute_func = lambda sig, lag: compute_EDMD(sig, lag, lamb=lamb, dictionary_method=dictionary_method, max_order=max_order, grouped_by=grouped_by, n_clusters=n_clusters, include_signal=include_signal, normalize=normalize)
        predict_func = lambda sig, coefs, p, intercept: predict_EDMD(sig, coefs, p=p, intercept=intercept, max_order=max_order, grouped_by=grouped_by, include_signal=include_signal, normalize=normalize, predict_type='actual', persistence_baseline=True)
    else:
        raise ValueError(f"Method is {method} but must be one of ['VAR(p)', 'EDMD']")
    
    return compute_func, predict_func

def perform_stability_analysis(data, windows, max_lag, method='VAR(p)', T_pred=25, num_window_samples=5, dt=0.001, lamb=0, use_lamb_for_full_results=False, dictionary_method='hermite', max_order=-1, grouped_by=None, n_clusters=100, include_signal=False, normalize=False, verbose=False):
    lags = np.arange(1, max_lag + 1)

    compute_func, predict_func = get_method_funcs(method, lamb, dictionary_method, max_order, grouped_by, n_clusters, include_signal, normalize)
    
    grid_search_df = []
    iterator = tqdm(total = len(windows)*len(lags), disable=not verbose)
    for window in windows:
        stride = window
        min_ind = int(0/stride)
        max_ind = int((data.shape[0]*dt - window - T_pred*dt)/stride)
        possible_inds = np.arange(min_ind, max_ind + 1)
        window_inds = np.random.choice(possible_inds, size=(np.min([num_window_samples, len(possible_inds)])), replace=False)
    #     num_windows = int(np.floor((data.shape[0]-window)/stride)+1)
        for p in lags:
    #         for i in range(num_windows):
            for i in window_inds:
                start_ind = i*int(stride/dt)
                start_time = i*stride
                end_ind = i*int(stride/dt) + int(window/dt)
                end_time = i*stride + window
                if end_ind + T_pred <= data.shape[0]:

                    window_data = data[start_ind:end_ind]
                    results = compute_func(window_data, p)
                    test_data = data[end_ind - p:end_ind + T_pred]

                    train_prediction, train_true_vals, pb_mse_train, _ = predict_func(window_data, results['coefs'], p, results['intercept'])
                    train_mse = ((train_prediction - train_true_vals)**2).mean()
                    test_prediction, test_true_vals, pb_mse_test, _ = predict_func(test_data, results['coefs'], p, results['intercept'])
    #                 print(test_prediction.shape)
                    test_mse = ((test_prediction - test_true_vals)**2).mean()
                    # persistence_baseline = ((data[end_ind:end_ind + T_pred] - data[end_ind - 1:end_ind + T_pred -1])**2).mean()

                    # ADD TO DICTIONARY
                    results['train_mse'] = train_mse
                    results['test_mse'] = test_mse
                    # results['persistence_baseline'] = persistence_baseline
                    results['pb_mse_train'] = pb_mse_train
                    results['pb_mse_test'] = pb_mse_test

                    # ADD TIMESTAMPS
                    results['start_ind'] = start_ind
                    results['end_ind'] = end_ind
                    results['start_time'] = start_time
                    results['end_time'] = end_time

                    # ADD PARAMETERS
                    results['window'] = window
                    results['stride'] = stride
                    results['p'] = p
                    results['T_pred'] = T_pred

                    grid_search_df.append(results)
            iterator.update()
    iterator.close()
    grid_search_df = pd.DataFrame(grid_search_df)
    
    test_mse_mat = np.zeros((len(windows), len(lags)))
    for i, window in enumerate(windows):
        for j, p in enumerate(lags):
            test_mse_mat[i, j] = grid_search_df[np.logical_and(grid_search_df.window == window, grid_search_df.p == p)].test_mse.mean()
    
    def pick_2d_optimum(mat, thresh=0.95):
        true_min = mat.min()
        i_vals, j_vals = np.where(mat*thresh - true_min <= 0)
        selected_i = np.min(i_vals)
        selected_j = np.min(j_vals[i_vals == selected_i])
        selected_i, selected_j

        return selected_i, selected_j
    thresh = 0.9
    w_ind, p_ind = pick_2d_optimum(test_mse_mat, thresh)
    
    window = windows[w_ind]
    p = lags[p_ind]
#     p = 1

    if use_lamb_for_full_results:
        lamb_full = lamb
    else:
        lamb_full = 0
    full_results = get_stability_results(data, window, p, method=method, T_pred=T_pred, dt=dt, lamb=lamb_full, dictionary_method=dictionary_method, max_order=max_order, grouped_by=grouped_by, n_clusters=n_clusters, include_signal=include_signal, normalize=normalize, verbose=verbose)
    
    return full_results, grid_search_df, test_mse_mat, window, p

def get_stability_results(data, window, p, method='VAR(p)', T_pred=25, dt=0.001, lamb=0, dictionary_method='hermite', max_order=-1, grouped_by=None, n_clusters=100, include_signal=False, normalize=False, verbose=False):
    
    compute_func, predict_func = get_method_funcs(method, lamb, dictionary_method, max_order, grouped_by, n_clusters, include_signal, normalize)
    
    full_results = []
    stride = window
    num_windows = int(np.floor((data.shape[0]-int(window/dt))/int(stride/dt)+1))
    for i in tqdm(range(num_windows), disable=not verbose):
        start_ind = i*int(stride/dt)
        start_time = i*stride
        end_ind = i*int(stride/dt) + int(window/dt)
        end_time = i*stride + window
        if end_ind + T_pred <= data.shape[0]:

            window_data = data[start_ind:end_ind]
            results = compute_func(window_data, p)
            test_data = data[end_ind - p:end_ind + T_pred]

            train_prediction, train_true_vals, pb_mse_train, _ = predict_func(window_data, results['coefs'], p, results['intercept'])
            train_mse = ((train_prediction - train_true_vals)**2).mean()
            test_prediction, test_true_vals, pb_mse_test, _ = predict_func(test_data, results['coefs'], p, results['intercept'])
    #                 print(test_prediction.shape)
            test_mse = ((test_prediction - test_true_vals)**2).mean()
            # persistence_baseline = ((data[end_ind:end_ind + T_pred] - data[end_ind - 1:end_ind + T_pred -1])**2).mean()

            # ADD TO DICTIONARY
            results['train_mse'] = train_mse
            results['test_mse'] = test_mse
            # results['persistence_baseline'] = persistence_baseline
            results['pb_mse_train'] = pb_mse_train
            results['pb_mse_test'] = pb_mse_test

            # ADD TIMESTAMPS
            results['start_ind'] = start_ind
            results['end_ind'] = end_ind
            results['start_time'] = start_time
            results['end_time'] = end_time

            # ADD PARAMETERS
            results['window'] = window
            results['stride'] = stride
            results['p'] = p
            results['T_pred'] = T_pred

            full_results.append(results)
    full_results = pd.DataFrame(full_results)
    
    return full_results