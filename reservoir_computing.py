import itertools as it
from math import comb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

class ReservoirDS:
    def __init__(self, u, dt=1, D_r=300, d=6, d_tolerance=0.01, rho=1.2, beta=0, sigma=0.1,
                    a_vals=np.linspace(0, 2, 100), squared_unit_input_dims=[], squared_inds=None, r_init=None,
                                                                                        var_names=None):
        # inputs
        self.u = u # input time series --> (number of time steps, number of dimensions)
        self.D_r = D_r # number of reservoir nodes
        self.dt = dt # s, time step
        self.d = d # average degree of Erdos-Renyi network
        self.d_tolerance = d_tolerance # tolerance on the average degree of the network (from d)
        self.rho = rho # spectral radius of the adjacency matrix
        self.beta = beta # regularization parameter
        self.sigma = sigma # network weights are uniform [-sigma, sigma]
        self.a_vals = a_vals # network weights are uniform [-a, a] - this input is a list of values to try to obtain
                             # the desired degree of the network
        # self.squared_unit_input_dims = squared_unit_input_dims # the dimensions of the input to predict using partially
        #                                                        # squared reservoir activations (defaults to Dr/2)
        self.squared_inds = squared_inds # indices of the reservoir to square - default is to square all even rows
        if self.squared_inds is None:
            self.squared_inds = np.arange(D_r)[np.arange(D_r) % 2 == 0]
        self.r_init = r_init # initialization to the reservoir (must have shape (D_r,), defaults to zeros)
        if self.r_init is None:
            # r_init = np.random.randn(self.D_r)*0.01
            self.r_init = np.zeros(self.D_r)
        self.var_names = var_names # names of the variables - must have length D (dimension of input u)

        # compute further variables
        self.T = u.shape[0]*dt # s, duration of input time series
        self.D = u.shape[1] # dimension of input u
        indices = np.arange(D_r)
        np.random.shuffle(indices)
        W_in = np.zeros((D_r, self.D))
        for i, idx in enumerate(indices):
            u_dim = int(i/(D_r/self.D))
            W_in[idx, u_dim] = np.random.uniform(low=-sigma, high=sigma)
        self.W_in = W_in

        # placeholders
        self.avg_degree = None # average degree of the adjacency graph
        self.a = None # network weights are uniform [-a, a]
        self.A = None # the adjacency matrix
        self.spectral_radius = None # the spectral radius of the adjacency matrix
        self.P = None # the linear readout matrix obtained through linear regression
        self.num_steps_train = None # number of training steps
        self.num_steps_test = None # number of test steps
        self.r_train = None # the activations from the training regime
        self.v_train = None # the network outputs from the training regime
        self.r_test = None  # the activations from the test regime
        self.v_test = None  # the network outputs from the test regime

    def fill_adjacency_matrix_weights(self, A_binary, a):
        A = np.zeros(A_binary.shape)
        for i, j in it.product(range(self.D_r), range(self.D_r)):
            if A_binary[i, j]:
                A[i, j] = np.random.uniform(low=-a, high=a)
        return A

    def build_connectivity(self, debug=False):
        # set up the adjacency matrix
        p = self.d * self.D_r / (2 * comb(self.D_r, 2))
        avg_degree = self.d + self.d_tolerance + 1
        adjacency_graph = None
        while np.abs(avg_degree - self.d) >= self.d_tolerance:
            adjacency_graph = nx.erdos_renyi_graph(self.D_r, p)
            avg_degree = np.sum([adjacency_graph.degree[i] for i in range(self.D_r)]) / self.D_r
        self.avg_degree = avg_degree
        if debug:
            print(f"The average degree of the network is {self.avg_degree:.2f}")

        A_binary = nx.linalg.graphmatrix.adj_matrix(adjacency_graph).toarray()

        spectral_radius = np.zeros(len(self.a_vals))
        A_mats = np.zeros((len(self.a_vals), self.D_r, self.D_r))
        if debug:
            iterator = tqdm(total=len(self.a_vals))
        else:
            iterator = None
        for i, a in enumerate(self.a_vals):
            A_mats[i] = self.fill_adjacency_matrix_weights(A_binary, a)
            eigenvalues, _ = np.linalg.eig(A_mats[i])
            spectral_radius[i] = np.abs(eigenvalues).max()
            if debug:
                iterator.update()
        iterator.close()

        opt_ind = np.argmin(np.abs(spectral_radius - self.rho))
        self.a = self.a_vals[opt_ind]
        self.A = A_mats[opt_ind]
        self.spectral_radius = spectral_radius[opt_ind]

        if debug:
            plt.scatter(self.a, self.spectral_radius, c='red', zorder=2)
            plt.plot(self.a_vals, spectral_radius, zorder=1)

            plt.xlabel("Value of $a$ (maximum weight)")
            plt.ylabel("Spectral Radius")
            plt.title(rf"Spectral Radius Over Maximum Weight Values ($\rho = ${self.rho})")
            plt.show()

            print(f"The spectral radius is {self.spectral_radius:.3f}")

    def create_r_tilde(self, r):
        if self.squared_inds is not None:
            r_tilde = r.copy()
            if r.ndim == 2:
                r_tilde[:, self.squared_inds] = r[:, self.squared_inds] ** 2
            else: # r.ndim == 1
                r_tilde[self.squared_inds] = r[self.squared_inds] ** 2
        else:
            r_tilde = np.zeros(self.D_r)

        return r_tilde

    def W_out(self, r):
        # r_tilde = self.create_r_tilde(r)
        #
        # # pass vectors through P, using r_tilde if the dimension is in by squared_unit_input_dims
        # out_vectors = []
        # for i in range(self.D):
        #     if i in self.squared_unit_input_dims:
        #         out_vectors.append(self.P[i] @ r_tilde.T)
        #     else:
        #         out_vectors.append(self.P[i] @ r.T)
        #
        # return np.array(out_vectors).T

        # return (self.P @ r.T).T

        X = r.copy()
        if r.ndim == 2:
            X[:, self.squared_inds] = r[:, self.squared_inds]**2
        else: # r.ndim == 1
            X[self.squared_inds] = r[self.squared_inds] ** 2

        return (self.P @ X.T).T

    # use MAP estimate of matrix to minimize the linear regression loss function
    # u is (num_time_steps, D) and r is (num_time_steps, D_r)
    def regress_output_weights(self, u, r):
        # P = np.zeros((self.D, self.D_r))
        #
        # r_tilde = self.create_r_tilde(r)
        #
        # for i in range(self.D):
        #     if i in self.squared_unit_input_dims:
        #         P[i] = np.linalg.inv(r_tilde.T @ r_tilde + self.beta * np.eye(self.D_r)) @ r_tilde.T @ u[:, i]
        #     else:
        #         P[i] = np.linalg.inv(r.T @ r + self.beta * np.eye(self.D_r)) @ r.T @ u[:, i]
        #
        # return P

        # P = (u.T @ np.linalg.pinv(r.T))
        #
        # return P

        X = r.copy()
        X[:, self.squared_inds] = r[:, self.squared_inds] ** 2
        X[self.squared_inds] = r[self.squared_inds] ** 2

        return u.T @ X @ np.linalg.inv(X.T @ X + self.beta*np.eye(self.D_r))

    def train_and_test(self, percent_training_data=0.8, debug=False):
        # ============
        # TRAIN
        # ============

        num_steps_train = int(self.u.shape[0] * percent_training_data)
        num_steps_test = self.u.shape[0] - num_steps_train
        if self.A is None:
            self.build_connectivity()
        # u[t] ----> r[t+1] ---> v[t + 1] = u[t + 1]
        r = np.zeros((num_steps_train, self.D_r))
        r[0] = self.r_init

        for t in range(num_steps_train - 1):
            r[t + 1] = np.tanh(self.A @ r[t] + self.W_in @ self.u[t])

        self.P = self.regress_output_weights(self.u[1:num_steps_train], r[1:])
        self.r_train = r
        self.v_train = self.W_out(r)
        self.num_steps_train = num_steps_train

        if debug:
            self.print_train_results()

        # ============
        # TEST
        # ============

        r = np.zeros((num_steps_test, self.D_r))
        v_out = np.zeros((num_steps_test, 3))

        r[0] = np.tanh(self.A @ self.r_train[-1] + self.W_in @ self.v_train[-1])
        v_out[0] = self.W_out(r[0])

        for t in range(num_steps_test - 1):
            r[t + 1] = np.tanh(self.A @ r[t] + self.W_in @ v_out[t])
            v_out[t + 1] = self.W_out(r[t + 1])

        self.r_test = r
        self.v_test = v_out
        self.num_steps_test = num_steps_test

        if debug:
            self.print_test_results()

    def print_train_results(self):
        fig, axs = plt.subplots(self.D, 1, sharex='all')

        # for i, ax in enumerate(axs):
        #     ax.plot(np.arange(1, self.u.shape[0]) * self.dt, self.u[1:, i], label='actual')
        #     ax.plot(np.arange(1, self.u.shape[0]) * self.dt, self.v_train[1:, i], label='predicted')
        #     ax.set_ylabel(['x', 'y', 'z'][i])

        for i, ax in enumerate(axs):
            ax.plot(np.arange(1, self.num_steps_train) * self.dt, self.u[1:self.num_steps_train, i], label='actual')
            ax.plot(np.arange(1, self.num_steps_train) * self.dt, self.v_train[1:, i], label='predicted')
            ax.set_ylabel(self.var_names[i])

        axs[0].legend()
        plt.xlabel('Time (s)')
        plt.suptitle('Training Results')
        plt.show()

    def print_test_results(self):
        fig, axs = plt.subplots(self.D, 1, sharex='all')

        # for i, ax in enumerate(axs):
        #     ax.plot(np.arange(1, self.r_test.shape[0]) * self.dt, self.u[1:self.r_test.shape[0], i], label='actual')
        #     ax.plot(np.arange(1, self.r_test.shape[0]) * self.dt, self.v_test[1:, i], label='predicted')
        #     ax.set_ylabel(['x', 'y', 'z'][i])

        for i, ax in enumerate(axs):
            ax.plot(np.arange(self.num_steps_test) * self.dt, self.u[self.num_steps_train:, i], label='actual')
            ax.plot(np.arange(self.num_steps_test) * self.dt, self.v_test[:, i], label='predicted')
            ax.set_ylabel(self.var_names[i])

        axs[0].legend()
        plt.xlabel('Time (s)')
        plt.suptitle('Test Results')
        plt.show()

    # def test(self, T_test):
    #     if self.A is None:
    #         self.build_connectivity()
    #     # if self.P is None:
    #     if self.P1 is None:
    #         self.train()
    #     num_steps_test = int(T_test / self.dt)
    #     r = np.zeros((num_steps_test, self.D_r))
    #     v_out = np.zeros((num_steps_test, 3))
    #
    #     r[0] = self.r_init
    #     v_out[0] = self.W_out(r[0])
    #     r[1] = np.tanh(self.A @ r[0] + self.W_in @ self.u[0])
    #     v_out[1] = self.W_out(r[1])
    #
    #     for t in range(1, num_steps_test - 1):
    #         r[t + 1] = np.tanh(self.A @ r[t] + self.W_in @ v_out[t])
    #         v_out[t + 1] = self.W_out(r[t + 1])
    #
    #     self.r_test = r
    #     self.v_test = v_out

    # def train(self):
    #     if self.A is None:
    #         self.build_connectivity()
    #     # u[t] ----> r[t+1] ---> v[t + 1] = u[t + 1]
    #     r = np.zeros((self.u.shape[0], self.D_r))
    #     r[0] = self.r_init
    #
    #     for t in range(self.u.shape[0] - 1):
    #         r[t + 1] = np.tanh(self.A @ r[t] + self.W_in @ self.u[t])
    #
    #     # self.P = self.regress_output_weights(self.u[1:], r[1:])
    #     self.P1, self.P2 = self.regress_output_weights(self.u[1:], r[1:])
    #     self.r_train = r
    #     self.v_train = self.W_out(r)