import numpy as np
import dill
from nmjmc.nn import NeuralMJMCNetwork
from tqdm import tqdm, tqdm_notebook


class MJMCSampler:
    def __init__(
        self, nn, local_probability, energy_function, noise=0.03, dim=76, beta=1.0
    ):
        """

        Parameters
        ----------
        kernel                  neural network
        local_probability       probability to select local moves
        energy_function         energy of a batch of configurations
        noise                   noise for local steps
        dim                     dimensionality of the system
        """
        self.nn = nn
        self.local_probability = local_probability
        self.energy_function = energy_function
        self.noise = noise
        self.dim = dim
        self.N = dim // 2
        self.beta = beta

    def run(self, X0, nsteps, stride=1, reassign_labels=None, verbose=0, reporter=None):
        n_configs = X0.shape[0]
        X = np.zeros((nsteps // stride, n_configs, self.dim))
        E_traj = np.zeros((nsteps // stride, n_configs))
        X[0] = X0
        beta = self.beta
        y = np.zeros((n_configs, self.dim))
        x = np.copy(X0)
        E = self.energy_function(x)
        E_traj[0] = E
        pacc = np.zeros(n_configs)
        E_new = np.zeros(n_configs)
        pacc_mean_local = np.zeros(nsteps)
        pacc_mean_global = np.zeros(nsteps)
        max_index = (nsteps // stride) * stride
        iterator = range(0, max_index)

        if reporter == "notebook":
            iterator = tqdm_notebook(iterator)
        else:
            iterator = tqdm(iterator)
        for i in iterator:

            randoms = np.random.rand(n_configs)
            local_idcs = np.where(randoms < self.local_probability)[0]
            global_idcs = np.where(randoms > self.local_probability)[0]

            if not len(local_idcs) == 0:
                (
                    y[local_idcs],
                    pacc[local_idcs],
                    E_new[local_idcs],
                ) = self.perform_local_step(x[local_idcs], E[local_idcs], beta=beta)
                pacc_mean_local[i] = np.mean(pacc[local_idcs])
            if not len(global_idcs) == 0:
                (
                    y[global_idcs],
                    pacc[global_idcs],
                    E_new[global_idcs],
                ) = self.perform_global_step(x[global_idcs], E[global_idcs], beta=beta)
                pacc_mean_global[i] = np.mean(pacc[global_idcs])

            rands = np.random.rand(n_configs)
            accepted = np.where(rands < pacc)[0]
            rejected = np.where(rands >= pacc)[0]
            if reassign_labels is not None:
                y_accepted = reassign_labels(y[accepted])
            else:
                y_accepted = y[accepted]
            x[accepted] = np.copy(y_accepted)
            E[accepted] = E_new[accepted]
            if not i % stride:
                index = i // stride
                X[index] = np.copy(x)
                E_traj[index] = np.copy(E)

        if verbose == 0:
            return X
        if verbose >= 2:
            return X, np.mean(pacc_mean_local), np.mean(pacc_mean_global), E
        if verbose >= 1:
            return X, np.mean(pacc_mean_local), np.mean(pacc_mean_global)

    def perform_local_step(self, X, E, beta):
        n = X.shape[0]
        y = X + np.random.randn(n, self.dim) * self.noise
        E_new = self.energy_function(y)
        pacc = np.exp(-beta * (E_new - E))
        pacc = np.minimum(pacc, 1.0)
        return y, pacc, E_new

    def perform_global_step(self, X, E, beta):
        kernel_output = self.nn.generate_output(X)
        pacc, E_y = self.pacc_mh_np(kernel_output, E_x=E, beta=beta)
        pacc = np.minimum(pacc, 1.0)
        return kernel_output["y"], pacc, E_y

    def pacc_mh_np(
        self, x, w=None, y=None, sigma_x=None, z=None, sigma_y=None, E_x=None, beta=1
    ):
        # got input x as dictionary
        if isinstance(x, dict):
            pred = x
            x, y, j_x = pred["x"], pred["y"], pred["j_x"]
        # compute energy difference
        if E_x is None:
            E_x = self.energy_function(x)
        E_y = self.energy_function(y)
        dE = E_y - E_x
        log_j_x = np.sum(j_x, axis=1)
        pacc = np.exp(-beta * dE + log_j_x)
        return pacc, E_y


class VoronoiMixture:
    def __init__(
        self,
        cluster_centers,
        kernels,
        selection_probabilities,
        energy_function,
        kernel_connectivity,
        noise=0.03,
        dim=76,
        beta=1.0,
    ):
        """

        Parameters
        ----------
        cores                   np array of shape ((coords, radii), ncores, dim)
        kernels                 list of kernels (instances of networks)
        selection_probabilities 2d np array p_ij is probability to select kernel j given in core i
        energy_function         energy of a batch of configurations
        kernel_connectivity     list of connections that kernels provide in terms of core indices
        noise                   noise for local steps
        dim                     dimensionality of the system
        """
        assert np.all(
            np.sum(selection_probabilities, axis=1) == 1.0
        ), "Selection probability doesnt add to one."
        self.cluster_centers = cluster_centers
        self.kernels = kernels
        # add 1 for the local kernel
        self.n_kernels = len(kernels) + 1
        self.n_cores = len(cluster_centers) + 1
        assert selection_probabilities.shape[0] == self.n_cores, (
            "First axis of selection_probabilities must match " "number of cores + 1"
        )
        assert selection_probabilities.shape[1] == self.n_kernels, (
            "Second axis of selection_probabilities must match" "number of kernels + 1"
        )
        self.selection_probabilities = selection_probabilities
        self.energy_function = energy_function
        self.noise = noise
        self.dim = dim
        self.N = dim // 2
        self.beta = beta
        # self.dimer_split = 1.65
        self.kernel_connectivity = np.array(kernel_connectivity)
        # kernel connectivity gives the set of cores that are connected by a kernel
        self.kernel_connectivity_matrix = -np.ones(
            (self.n_kernels, self.n_cores), np.int
        )
        for kernel, core in enumerate(kernel_connectivity):
            self.kernel_connectivity_matrix[kernel, core[0]] = core[1]
            self.kernel_connectivity_matrix[kernel, core[1]] = core[0]
        # selection ratios gives the ratio of probabilities in either selecting one kernel from either of its connected
        # cores vs the other one
        self.selection_ratios = np.zeros((self.n_kernels, self.n_cores))
        for kernel, core in enumerate(kernel_connectivity):
            self.selection_ratios[kernel, core[0]] = (
                selection_probabilities[core[1], kernel]
                / selection_probabilities[core[0], kernel]
            )
            self.selection_ratios[kernel, core[1]] = (
                selection_probabilities[core[0], kernel]
                / selection_probabilities[core[1], kernel]
            )
        self.local_selection_ratios = np.eye(self.n_cores)
        for i in range(self.n_cores):
            self.local_selection_ratios[i, -1] = 1.0 / selection_probabilities[i, -1]
            self.local_selection_ratios[-1, i] = selection_probabilities[i, -1]
        self.pacc_mh_np = self._pacc_gaussian_density
        self.output_key = "y_noisy"
        if isinstance(self.kernels[0], NeuralMJMCNetwork):
            self.pacc_mh_np = self._pacc_deterministic
            self.output_key = "y"

    def run(self, X0, nsteps, reassign_labels=None, verbose=1, reporter=None):
        n_configs = X0.shape[0]
        X = np.zeros((nsteps, n_configs, self.dim))
        pacc_traj = np.zeros((nsteps - 1, n_configs))
        kernel_traj = np.zeros((nsteps - 1, n_configs))
        core_traj = np.zeros((nsteps - 1, n_configs))
        E = np.zeros((nsteps, n_configs))
        X[0] = X0
        E[0] = self.energy_function(X[0])
        beta = self.beta
        y = np.zeros((n_configs, self.dim))
        pacc = np.zeros(n_configs)
        E_new = np.zeros(n_configs)
        if reporter == "notebook":
            iterator = tqdm_notebook(range(0, nsteps - 1))
        else:
            iterator = tqdm(range(0, nsteps - 1))
        for i in iterator:
            selected_kernel, cores = self.select_kernel(X[i])
            core_traj[i] = cores
            kernel_traj[i] = selected_kernel
            local_idcs = np.where(selected_kernel == self.n_kernels - 1)[0]
            # print(selected_kernel, cores)
            if not len(local_idcs) == 0:
                (
                    y[local_idcs],
                    pacc[local_idcs],
                    E_new[local_idcs],
                ) = self.perform_local_step(
                    X[i, local_idcs], E[i, local_idcs], cores[local_idcs], beta=beta
                )
                pacc_traj[i, local_idcs] = pacc[local_idcs]
            for j in range(0, self.n_kernels - 1):
                kernel_idcs = np.where(selected_kernel == j)[0]
                # exclude the case that certain kernel was never selected in that step
                if not len(kernel_idcs) == 0:
                    (
                        y[kernel_idcs],
                        pacc[kernel_idcs],
                        E_new[kernel_idcs],
                    ) = self.perform_global_step(
                        X[i, kernel_idcs], j, cores[kernel_idcs], beta=beta
                    )
                    pacc_traj[i, kernel_idcs] = pacc[kernel_idcs]
            rands = np.random.rand(n_configs)
            accepted = np.where(rands < pacc)[0]
            rejected = np.where(rands >= pacc)[0]
            if reassign_labels is not None:
                y_accepted = reassign_labels(y[accepted])
            else:
                y_accepted = y[accepted]
            X[i + 1, accepted] = y_accepted
            E[i + 1, accepted] = E_new[accepted]
            X[i + 1, rejected] = X[i, rejected]
            E[i + 1, rejected] = E[i, rejected]
        mean_pacc = np.zeros((self.n_kernels, 2))
        for i in range(self.n_kernels - 1):
            for direction in range(2):
                idcs = np.where(
                    np.logical_and(
                        core_traj.flatten() == self.kernel_connectivity[i][direction],
                        kernel_traj.flatten() == i,
                    )
                )[0]
                mean_pacc[i, direction] = np.mean(pacc_traj.flatten()[idcs])
        idcs_local = np.where(kernel_traj.flatten() == self.n_kernels - 1)[0]
        if verbose == 0:
            return X
        mean_pacc[-1, 0] = np.mean(pacc_traj.flatten()[idcs_local])
        if verbose >= 3:
            return X, mean_pacc, E, core_traj, pacc_traj
        if verbose >= 2:
            return X, mean_pacc, E
        if verbose >= 1:
            return X, mean_pacc

    def perform_local_step(self, X, E, cores, beta):
        n = X.shape[0]
        y = X + np.random.randn(n, self.dim) * self.noise
        E_new = self.energy_function(y)
        pacc = np.exp(-beta * (E_new - E))
        cores_y = self.assign_core(y)
        selection_ratio = self.local_selection_ratios[cores, cores_y]
        # selection_ratio = 1
        pacc = np.minimum(selection_ratio * pacc, 1.0)
        return y, pacc, E_new

    def perform_global_step(self, X, kernel_id, core_id, beta):
        kernel = self.kernels[kernel_id]
        kernel_output = kernel.generate_output(X)
        target_kernel = self.kernel_connectivity_matrix[kernel_id, core_id]
        pacc = self.within_core(
            kernel_output[self.output_key], target_kernel
        ) * self.pacc_mh_np(kernel_output, beta=beta)
        selection_ratio = self.selection_ratios[kernel_id, core_id]
        pacc = np.minimum(selection_ratio * pacc, 1.0)
        return (
            kernel_output[self.output_key],
            pacc,
            self.energy_function(kernel_output[self.output_key]),
        )

    def select_kernel(self, X):
        cores = self.assign_core(X)
        selected_kernel = np.zeros(len(X))
        for i in range(len(X)):
            if cores[i] == self.n_cores - 1:
                # select global kernel with probability=1 in this case, as no other is available
                selected_kernel[i] = self.n_kernels - 1
            else:
                # randomly select one kernel from the kernel probability matrix
                p = self.selection_probabilities[cores[i]]
                selected_kernel[i] = np.random.choice(self.n_kernels, 1, p=p)
        return selected_kernel, cores

    def assign_core(self, X):
        xcomp = X[:, ::2]
        xcomp = np.expand_dims(xcomp, 1)
        ycomp = X[:, 1::2]
        ycomp = np.expand_dims(ycomp, 1)
        batchsize = len(xcomp)
        center_distance = (xcomp - self.cluster_centers[:, ::2]) ** 2 + (
            ycomp - self.cluster_centers[:, 1::2]
        ) ** 2
        core_assignment = np.argmin(center_distance, axis=1).reshape(-1)
        return core_assignment

    def within_core(self, X, core_id):
        # this function does not check if any x is in the global core!
        assigned_core = self.assign_core(X)
        return assigned_core == core_id

    # def _default_split_function(self, X):
    #     return np.linalg.norm(X[:, :2] - X[:, 2:4], axis=1) > self.dimer_split

    def _pacc_gaussian_density(
        self, x, w=None, y=None, sigma_x=None, z=None, sigma_y=None, beta=1
    ):
        # got input x as dictionary
        if isinstance(x, dict):
            pred = x
            x, w, y, sigma_x, z, sigma_y = (
                pred["x"],
                pred["w"],
                pred["y_noisy"],
                pred["sigma_x"],
                pred["z"],
                pred["sigma_y"],
            )
        # compute proposal probabilities
        log_pprop_xy = -np.sum(w ** 2, axis=1) / 2.0 - np.sum(np.log(sigma_x), axis=1)
        w_y = (x - z) / sigma_y
        log_pprop_yx = -np.sum(w_y ** 2, axis=1) / 2.0 - np.sum(np.log(sigma_y), axis=1)
        # compute energy difference
        dE = (
            self.energy_function(y)
            - self.energy_function(x)
            + log_pprop_xy
            - log_pprop_yx
        )
        pacc = np.exp(-beta * dE)
        return pacc

    def _pacc_deterministic(self, x, beta=1):
        # got input x as dictionary
        if isinstance(x, dict):
            pred = x
            x, y, log_j_x = pred["x"], pred["y"], pred["j_x"]
        log_j_x = np.sum(log_j_x, axis=1)
        dE = self.energy_function(y) - self.energy_function(x)
        pacc = np.exp(-beta * dE + log_j_x)
        return pacc

    def save(self, filename):
        for i in range(len(self.kernels)):
            self.kernels[i].save_network(filename + "_kernel_" + str(i))
        model_parameters = {
            "cores": self.cores,
            "selection_probabilities": self.selection_probabilities,
            "energy_function": self.energy_function,
            "kernel_connectivity": self.kernel_connectivity,
            "noise": self.noise,
            "dim": self.dim,
            "n_kernels": self.n_kernels - 1,
        }
        dill.dump(model_parameters, open(filename, "wb"))

    @classmethod
    def load(cls, filename):
        model_parameters = dill.load(open(filename, "rb"))
        kernels = []
        for i in range(model_parameters["n_kernels"]):
            nn = RNVPNetwork.load_network(filename + "_kernel_" + str(i))
            kernels.append(nn)
        model = cls(
            model_parameters["cores"],
            kernels,
            model_parameters["selection_probabilities"],
            model_parameters["energy_function"],
            model_parameters["kernel_connectivity"],
            model_parameters["noise"],
            model_parameters["dim"],
        )
        return model

    def run_return_time(self, x0, core_id, reassign_labels=None, max_iter=1e6):
        X = x0.reshape(1, self.dim)
        assert np.all(
            self.within_core(X, core_id)
        ), "At least one X0 does not start in core"
        E = self.energy_function(X)
        t = 0
        while self.within_core(X, core_id):
            t += 1
            y, pacc, E_new = self.perform_local_step(X, E, core_id)
            if np.random.rand(1) < pacc:
                X = y
                E = E_new
        print("exited after {} steps".format(t))
        t = 0
        while not self.within_core(X, core_id) and t < max_iter:
            t += 1.0
            y, pacc, E_new = self.perform_local_step(X, E, 2)
            if np.random.rand(1) < pacc:
                E = E_new
                if reassign_labels is not None:
                    X = reassign_labels(y)
                else:
                    X = y
            if t == max_iter - 2:
                print("max iterations reached")
                np.save("max_iter_reached", X)
        return t
