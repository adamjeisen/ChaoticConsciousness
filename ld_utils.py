import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from scipy.linalg import svdvals
from tqdm.auto import tqdm

# returns the eigenvalues of the VAR matrix for each window of data in the time series
# also, optionally, returns the covariance matrix of the white noise for each window
# dt, window, stride are in seconds
# pts has shape (num_time_steps, num_dims)
def compute_eigs(pts, dt, window=1, stride=None, return_sigma_norms=False, return_A=False, PCA_dim=-1, verbose=False):
    # -------------------
    # Initialize Variables
    # -------------------
    n = pts.shape[0]
    k = int(window/dt) # window size in time steps
    # default to window stride
    if stride is None:
        r = 0 # overlap size in time steps
    else:
        r = int((window -
         stride)/dt) # overlap size in time steps
    
    num_win = int(np.floor((n - k)/(k - r) + 1))
    
    # -------------------
    # Initialize Results Arrays
    # -------------------
    results = {}
    if PCA_dim > 0:
        if PCA_dim < 2:
            raise ValueError(f"PCA dimension must be greater than 1; provided value was {PCA_dim}")
        results['eigs'] = np.zeros((num_win, PCA_dim))
        pca = PCA(n_components=PCA_dim)
        results['explained_variance'] = np.zeros((num_win, PCA_dim))
        if return_A:
            results['A_mat'] = np.zeros((num_win, PCA_dim, PCA_dim))
            results['A_mat_with_bias'] = np.zeros((num_win, PCA_dim + 1, PCA_dim))
        # results['params'] = np.zeros((num_win, PCA_dim + 1, PCA_dim)) # to get coefs, params[1:].T
    else:
        results['eigs'] = np.zeros((num_win, pts.shape[1]))
        if return_A:
            results['A_mat'] = np.zeros((num_win, pts.shape[1], pts.shape[1]))
            results['A_mat_with_bias'] = np.zeros((num_win, pts.shape[1] + 1, pts.shape[1]))
        # results['params'] = np.zeros((num_win, pts.shape[1] + 1, pts.shape[1])) # to get coefs, params[1:].T
    
    if return_sigma_norms:
        results['sigma_norms'] = np.zeros(num_win)
    results['sigma2_ML'] = np.zeros(num_win)
    results['AIC'] = np.zeros(num_win)

    results['window_locs'] = np.zeros((num_win, 2), dtype=np.int)
    # results['1step_pred'] = np.zeros((num_win - 1, pts.shape[1]))
    # results['1step_pred_dist'] = np.zeros(num_win - 1)
    # results['mse'] = np.zeros(num_win - 1)

    # -------------------
    # Compute Eigenvalues
    # -------------------

    if verbose:
        iterator = tqdm(total = num_win)

    for i in range(num_win):
        results['window_locs'][i] = (i*(k - r), i*(k - r) + k)
        chunk = pts[results['window_locs'][i][0]:results['window_locs'][i][1]]
        if PCA_dim > 0:
            chunk = pca.fit_transform(chunk)
            results['explained_variance'][i] = pca.explained_variance_ratio_
        model = VAR(chunk)
        VAR_results = model.fit(1)
        if return_A:
            results['A_mat'][i] = VAR_results.coefs[0]
            results['A_mat_with_bias'][i] = VAR_results.params
        e,_ = np.linalg.eig(VAR_results.coefs[0])      
        results['eigs'][i] = np.abs(e)

        results['sigma2_ML'][i] = np.linalg.norm(VAR_results.endog[1:] - (VAR_results.endog_lagged @ VAR_results.params), axis=1).sum()/(k - 2)
        results['AIC'][i] = k*np.log(results['sigma2_ML'][i]) + 2

        # if i < num_win - 1:
        #     results['1step_pred'][i] = np.concatenate([[1], VAR_results.endog[-1]]) @ VAR_results.params
        #     results['1step_pred_dist'][i] = np.linalg.norm(pts[results['window_locs'][i][1]] - results['1step_pred'][i])
        #     results['mse'][i] = (results['1step_pred_dist'][i]**2)/pts.shape[1]
        # results['params'][i] = VAR_results.params

        if return_sigma_norms:
            results['sigma_norms'][i] = svdvals(VAR_results.sigma_u)[0]
        if verbose:
            iterator.update()
    if verbose:
        iterator.close()

    return results

# get the covariance of the data
# dt, window, stride are in seconds
# pts has shape (num_time_steps, num_dims)
def get_data_sigma_norms(pts, dt, window=1, stride=None, verbose=False):
    n = pts.shape[0]
    k = int(window/dt) # window size in time steps
    # default to window stride
    if stride is None:
        r = 0 # overlap size in time steps
    else:
        r = int((window - stride)/dt) # overlap size in time steps
    
    num_win = int(np.floor((n - k)/(k - r) + 1))
    
    sigma_norms = np.zeros(num_win)
    if verbose:
        iterator = tqdm(total = num_win)
    for i in range(num_win):
        chunk = pts[i*(k - r):i*(k - r) + k]
        sigma_norms[i] = svdvals(np.cov(chunk.T))[0]

        if verbose:
            iterator.update()
    
    if verbose:
        iterator.close()

    return sigma_norms