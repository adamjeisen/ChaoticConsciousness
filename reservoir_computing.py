import itertools as it
from math import comb
import matplotlib.pyplot as plt
import networkx as nx
from neural_analysis import spectra
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm

from nld_utils import lyap_spectrum_QR

class ReservoirDS:
    def __init__(self, u, dt=1, D_r=300, d=6, d_tolerance=0.01, p=None, rho=1.2, rho_tolerance=0.1,
                        beta=0, sigma=0.1, squared_inds=None, r_init=None, var_names=None,
                                                                                train_noise=False, test_noise=False):
        # inputs
        self.u = u # input time series --> (number of time steps, number of dimensions)
        self.D_r = D_r # number of reservoir nodes
        self.dt = dt # s, time step
        self.d = d # average degree of Erdos-Renyi network
        self.d_tolerance = d_tolerance  # tolerance on the average degree of the network (from d)
        self.p = p # probability of an edge in the Erdos-Renyi network - if not None, overrides d
        self.rho = rho # spectral radius of the adjacency matrix
        self.beta = beta # regularization parameter
        self.sigma = sigma # input network weights are uniform [-sigma, sigma]
        self.squared_inds = squared_inds # indices of the reservoir to square - default is to square all even rows
        if self.squared_inds is None:
            self.squared_inds = np.arange(D_r)[np.arange(D_r) % 2 == 0]
        self.r_init = r_init # initialization to the reservoir (must have shape (D_r,), defaults to zeros)
        if self.r_init is None:
            # r_init = np.random.randn(self.D_r)*0.01
            self.r_init = np.zeros(self.D_r)
        self.var_names = var_names # names of the variables - must have length D (dimension of input u)
        self.train_noise = train_noise # boolean, whether to include Gaussian noise on the training inputs
        self.test_noise = test_noise # boolean, whether to include Gaussian noise on the testing inputs

        # compute further variables
        self.T = u.shape[0]*dt # s, duration of input time series
        self.D = u.shape[1] # dimension of input u
        indices = np.arange(D_r)
        np.random.shuffle(indices)
        W_in = np.zeros((D_r, self.D))
        for i, idx in enumerate(indices):
            u_dim = int(i/(D_r/self.D))
            W_in[idx, u_dim] = np.random.uniform(low=-sigma, high=sigma)
        # W_in = np.random.randn(D_r, self.D)*(1/np.sqrt(self.D))
        self.W_in = W_in

        # placeholders
        self.avg_degree = None # average degree of the adjacency graph
        self.a = None # network weights are uniform [-a, a]
        self.A = None # the adjacency matrix
        self.P = None # the linear readout matrix obtained through linear regression
        self.num_steps_train = None # number of training steps
        self.num_steps_test = None # number of test steps
        self.r_train = None # the activations from the training regime
        self.v_train = None # the network outputs from the training regime
        self.r_test = None  # the activations from the test regime
        self.v_test = None  # the network outputs from the test regime
        self.power_spectra_true = None # power spectra from actual test sequence
        self.power_spectra_test = None # power spectra from predicted test sequence
        self.freqs = None # frequencies for power spectra
        self.Js = None # Jacobian matrices for the test regime
        self.lyaps = None # lyapunov spectrum for the reservoir

    def fill_adjacency_matrix_weights(self, A_binary, a):
        A = np.zeros(A_binary.shape)
        for i, j in it.product(range(self.D_r), range(self.D_r)):
            if A_binary[i, j]:
                A[i, j] = np.random.uniform(low=-a, high=a)
        return A

    def build_connectivity(self, debug=False):
        # set up the adjacency matrix
        if self.p is None:
            p = self.d * self.D_r / (2 * comb(self.D_r, 2))
            avg_degree = self.d + 1 + self.d_tolerance # the 1 ensures that the loop will run at least once
            adjacency_graph = None
            while np.abs(avg_degree - self.d) >= self.d_tolerance:
                adjacency_graph = nx.erdos_renyi_graph(self.D_r, p)
                avg_degree = np.sum([adjacency_graph.degree[i] for i in range(self.D_r)]) / self.D_r
        else:
            adjacency_graph = nx.erdos_renyi_graph(self.D_r, self.p)
            avg_degree = np.sum([adjacency_graph.degree[i] for i in range(self.D_r)]) / self.D_r
        self.avg_degree = avg_degree

        if debug:
            print(f"The average degree of the network is {self.avg_degree:.2f}")

        A_binary = nx.linalg.graphmatrix.adj_matrix(adjacency_graph).toarray()
        A = self.fill_adjacency_matrix_weights(A_binary, 1)
        A_s = sparse.csr_matrix(A)
        temp_radius = np.abs(sparse.linalg.eigs(A_s, k=1)[0][0])
        a = self.rho / temp_radius
        self.a = a
        self.A = a*A

    def W_out(self, r):
        X = r.copy()
        if r.ndim == 2:
            X[:, self.squared_inds] = r[:, self.squared_inds]**2
        else: # r.ndim == 1
            X[self.squared_inds] = r[self.squared_inds]**2

        return (self.P @ X.T).T

    # use MAP estimate of matrix to minimize the linear regression loss function
    # u is (num_time_steps, D) and r is (num_time_steps, D_r)
    def regress_output_weights(self, u, r):
        X = r.copy()
        if r.ndim == 2:
            X[:, self.squared_inds] = r[:, self.squared_inds] ** 2
        else: # r.ndim == 1
            X[self.squared_inds] = r[self.squared_inds]**2

        return u.T @ X @ np.linalg.inv(X.T @ X + self.beta*np.eye(self.D_r))

    def train_and_test(self, percent_training_data=0.8, train_noise=None, test_noise=None):
        # ============
        # TRAIN
        # ============

        num_steps_train = int(self.u.shape[0] * percent_training_data)
        if train_noise is None:
            train_noise = np.zeros((num_steps_train, self.D))
        num_steps_test = self.u.shape[0] - num_steps_train
        if test_noise is None:
            test_noise = np.zeros((num_steps_test, self.D))
        if self.A is None:
            self.build_connectivity()
        # u[t] ----> r[t+1] ---> v[t + 1] = u[t + 1]
        r = np.zeros((num_steps_train, self.D_r))
        r[0] = self.r_init

        for t in range(num_steps_train - 1):
            r[t + 1] = np.tanh(self.A @ r[t] + self.W_in @ (self.u[t] + train_noise[t]))

        self.P = self.regress_output_weights(self.u[1:num_steps_train], r[1:])
        self.r_train = r
        self.v_train = self.W_out(r)
        self.num_steps_train = num_steps_train

        # ============
        # TEST
        # ============

        r = np.zeros((num_steps_test, self.D_r))
        v_out = np.zeros((num_steps_test, self.D))

        r[0] = np.tanh(self.A @ self.r_train[-1] + self.W_in @ (self.v_train[-1] + test_noise[0]))
        v_out[0] = self.W_out(r[0])

        for t in range(num_steps_test - 1):
            r[t + 1] = np.tanh(self.A @ r[t] + self.W_in @ (v_out[t] + + self.test_noise*np.random.randn(self.D)))
            v_out[t + 1] = self.W_out(r[t + 1])

        self.r_test = r
        self.v_test = v_out
        self.num_steps_test = num_steps_test

    def compute_power_spectra(self, freq_range=np.array([0, 80]), normalize=True, spec_scale=100):
        freqs, _ = spectra.get_freq_sampling(1 / self.dt, spectra._next_power_of_2(self.num_steps_test),
                                                                                        freq_range=freq_range)
        num_freqs = len(freqs)

        power_spectra_true = np.zeros((self.D, num_freqs))
        power_spectra_test = np.zeros((self.D, num_freqs))

        for i in range(self.D):
            spec_true, _ = spectra.spectrum(self.u[self.num_steps_train:, i], smp_rate=1 / self.dt,
                                                     spec_type='power', freq_range=freq_range)
            spec_test, _ = spectra.spectrum(self.v_test[:, i], smp_rate=1 / self.dt, spec_type='power',
                                                     freq_range=freq_range)
            # normalize (to 0-1)
            if normalize:
                spec_true  = spec_scale*(spec_true - spec_true.min())/(spec_true.max() - spec_true.min())
                spec_test = spec_scale*(spec_test - spec_test.min()) / (spec_test.max() - spec_test.min())
            power_spectra_true[i] = spec_true
            power_spectra_test[i] = spec_test

        self.power_spectra_true = power_spectra_true
        self.power_spectra_test = power_spectra_test
        self.freqs = freqs

    def compute_jacobians(self, debug=False):
        def sech(x):
            return 1 / np.cosh(x)
        Js = np.zeros((self.num_steps_test, self.D_r, self.D_r))

        iterator = None
        if debug:
            print("Computing Jacobians")
            iterator = tqdm(total=self.num_steps_test)

        for t in range(-1, self.num_steps_test - 1):
            if t < 0:
                r = self.r_train[-1]
                v = self.v_train[-1]
            else:
                r = self.r_test[t]
                v = self.v_test[t]

            D1 = np.diag(sech(self.A @ r + self.W_in @ v) ** 2)
            W_feedback = self.W_in @ self.P
            W_feedback[:, self.squared_inds] *= 2 * r[self.squared_inds]
            D2 = self.A + W_feedback
            Js[t + 1] = D1 @ D2

            if debug:
                iterator.update()
        if debug:
            iterator.close()

        self.Js = Js

    def compute_lyap_spectrum_QR(self, debug=False):
        # if self.Js is None:
        #     self.compute_jacobians(debug)
        self.compute_jacobians(debug)
        Js = self.Js
        T = self.num_steps_test*self.dt

        lyaps = lyap_spectrum_QR(Js, T, debug=debug)
        lyaps.sort()
        lyaps = lyaps[::-1]
        self.lyaps = lyaps

    # =====================================================================
    # PLOTTING FUNCTIONS
    # =====================================================================

    def plot_train_and_test_results(self, num=None, indices=None, fig=None):
        if num is None and indices is None:
            num = self.D
            indices = np.arange(self.D)
        else:
            if indices is None:
                indices = np.random.choice(np.arange(self.D), size=(num,), replace=False)
            else:
                num = len(indices)

        if fig is None:
            fig = plt.figure(figsize=(12, 4))
        # subfigs = fig.subfigures(1, 2, width_ratios = [self.num_steps_train/(self.num_steps_test + self.num_steps_train),
        #                                         self.num_steps_test/(self.num_steps_test + self.num_steps_train)])

        train_width = self.num_steps_train/(self.num_steps_test + self.num_steps_train)
        test_width = self.num_steps_test/(self.num_steps_test + self.num_steps_train)
        axs = fig.subplots(num, 2, sharex='col', sharey='row',
                           gridspec_kw={'width_ratios': [train_width, test_width]})

        # ====================
        # TRAIN RESULTS
        # ====================
        # subfig = subfigs[0]
        # axs = subfig.subplots(num, 1, sharex='all')

        for i, ax in zip(indices, axs[:, 0]):
            ax.plot(np.arange(1, self.num_steps_train) * self.dt, self.u[1:self.num_steps_train, i], label='actual')
            ax.plot(np.arange(1, self.num_steps_train) * self.dt, self.v_train[1:, i], label='predicted')
            ax.set_ylabel(self.var_names[i])

        # axs[0].legend()
        # axs[-1].set_xlabel('Time (s)')
        # subfig.suptitle('Training Results')

        # ====================
        # TEST RESULTS
        # ====================
        # subfig = subfigs[1]
        # axs = subfig.subplots(num, 1, sharex='all')

        for i, ax in zip(indices, axs[:, 1]):
            ax.plot(np.array(np.arange(self.num_steps_test) + self.num_steps_train) * self.dt, self.u[self.num_steps_train:, i], label='actual')
            ax.plot(np.array(np.arange(self.num_steps_test) + self.num_steps_train) * self.dt, self.v_test[:, i], label='predicted')
            # ax.set_ylabel(self.var_names[i])

        axs[0][0].legend()
        axs[-1][0].set_xlabel('Time (s)')
        # subfig.suptitle('Training Results')

        if fig is None:
            plt.show()

    def plot_power_spectra(self, num=None, indices=None, fig=None):
        # if self.power_spectra_true is None:
        #     self.compute_power_spectra()

        self.compute_power_spectra()

        if num is None and indices is None:
            if fig is None:
                print_ = True
                fig = plt.figure(figsize=(12, 4))
            else:
                print_ = False

            axs = fig.subplots(1, 2)

            ax = axs[0]
            ax.set_title("Actual Spectra")
            im = ax.pcolormesh(self.power_spectra_true)
            fig.colorbar(im, ax=ax, label='Power')
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Input Dimension")
            xtick_locs = np.int0(ax.get_xticks()[:-1])
            ax.set_xticks(xtick_locs)
            ax.set_xticklabels([f"{val:.1f}" for val in self.freqs[xtick_locs]])

            ax = axs[1]
            ax.set_title("Predicted Spectra")
            im = ax.pcolormesh(self.power_spectra_test)
            fig.colorbar(im, ax=ax, label='Power')
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Input Dimension")
            ax.set_xticks(xtick_locs)
            ax.set_xticklabels([f"{val:.1f}" for val in self.freqs[xtick_locs]])

            fig.suptitle(f"Spectra MSE = {((self.power_spectra_true - self.power_spectra_test) ** 2).mean():.3f}")

            if print_:
                plt.tight_layout()
                plt.show()

        else:
            if indices is None:
                indices = np.random.choice(np.arange(self.D), size=(num,), replace=False)
            else:
                num = len(indices)

            if fig is None:
                print_ = True
                fig = plt.figure(figsize=(14, 4))
            else:
                print_ = False

            axs = fig.subplots(1, num)
            for i, ind in enumerate(indices):
                ax = axs[i]
                ax.plot(self.freqs, self.power_spectra_true[ind], label='actual')
                ax.plot(self.freqs, self.power_spectra_test[ind], label='predicted')
                ax.set_title(f"{self.var_names[ind]}, MSE = "
                             f"{((self.power_spectra_true[ind] - self.power_spectra_test[ind])**2).mean():.3f}")
                ax.set_xlabel('Freqs (Hz)')
                ax.set_ylabel('Power')
            ax.legend()

            if print_:
                plt.show()

    def plot_activity_and_outputs(self, fig=None):
        if fig is None:
            print_ = True
            fig = plt.figure(figsize=(14, 8))
            axs = plt.subplots(2, 3)
        else:
            print_ = False
            axs = fig.subplots(2, 3)

        ax = axs[0][0]
        im = ax.pcolormesh(self.r_train.T)
        fig.colorbar(im, ax=ax)
        ax.set_title("Train Activity")
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron #')

        ax = axs[0][1]
        ax.set_title("Test Activity")
        im = ax.pcolormesh(self.r_test.T)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron #')

        ax = axs[0][2]
        r2 = self.r_test.copy()
        r2[:, self.squared_inds] = r2[:, self.squared_inds] ** 2
        im = ax.pcolormesh(r2.T)
        fig.colorbar(im, ax=ax)
        ax.set_title("Squared Test Activity")
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron #')

        ax = axs[1][0]
        im = ax.pcolormesh(self.v_train.T)
        fig.colorbar(im, ax=ax)
        ax.set_title("Train Output")
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel("Output Index")

        ax = axs[1][1]
        im = ax.pcolormesh((self.P @ self.r_test.T))
        fig.colorbar(im, ax=ax)
        ax.set_title("Linear Test Output")
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel("Output Index")

        ax = axs[1][2]
        ax.set_title("Test Output (From Squared Units)")
        im = ax.pcolormesh(self.v_test.T)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel("Output Index")

        if print_:
            plt.tight_layout()
            plt.show()

    def plot_all(self, num=3):
        num = min(num, self.D)
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        # subfigs = fig.subfigures(4, 1, height_ratios=[1.3, 2, 1, 1])
        subfigs = fig.subfigures(3, 1, height_ratios=[1.3, 1, 1])
        indices = np.random.choice(np.arange(self.D), size=(num,), replace=False)

        # =================
        # TRAINING AND TESTING RESULTS
        # =================
        subfigs[0].suptitle('Train and Test Results')
        self.plot_train_and_test_results(indices=indices, fig=subfigs[0])
        # # =================
        # # PLOT NETWORK ACTIVITY AND OUTPUTS
        # # =================
        # subfigs[1].suptitle('Network Activity and Outputs')
        # self.plot_activity_and_outputs(fig=subfigs[1])

        # =================
        # PLOT POWER SPECTRA
        # =================
        self.plot_power_spectra(fig=subfigs[1])
        self.plot_power_spectra(indices=indices, fig=subfigs[2])

        plt.show()