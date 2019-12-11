import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sd
import tensorflow.keras as keras
from ._internal.hungarianMethod import linear_sum_assignment
from functools import partial

# use agg backend for allegro compatibility
# plt.switch_backend('agg')


def draw_mjmc(nn, energy, x, verbose=0, filename=None):
    # make sure x is numpy array
    x = np.array(x)
    n_samples = len(x)
    if len(x.shape) == 1:
        n_samples = 1
    pred = nn.model.predict(
        [
            x.reshape(n_samples, nn.dim),
            np.random.randn(n_samples, nn.dim).astype(np.float32),
        ]
    )
    pred_dict = nn.split_output(pred, return_dict=True)
    x = pred_dict["x"]
    y = pred_dict["y"]
    j_x = pred_dict["j_x"]
    log_j = np.sum(j_x)

    n_plots = 2

    fig, axs = plt.subplots(n_samples, n_plots, figsize=(n_plots * 5, n_samples * 5))
    if n_samples == 1:
        axs = [axs]
    for k, ax in enumerate(axs):
        draw_config(
            x[k],
            ax=ax[0],
            draw_labels=True,
            title="X energy: " + str(energy(x[k].reshape(1, nn.dim))),
        )
        draw_config(
            y[k],
            ax=ax[1],
            draw_labels=True,
            title="Y energy: " + str(energy(y[k].reshape(1, nn.dim))),
        )
        # draw_config(z[k], ax=ax[2], draw_labels=True, title='Z energy: '+str(energy(z[k].reshape(1, nn.dim))))
        if verbose > 0:
            delta_e = energy(y[k].reshape(1, nn.dim)) - energy(x[k].reshape(1, nn.dim))
            print("delta energy:", delta_e)
            print("log Jacobian", log_j)
            print("Pacc: ", np.exp(-delta_e + log_j))
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")


def draw_config(
    x,
    ax=None,
    dimercolor="blue",
    solventcolor="grey",
    boxcolor="grey",
    alpha=0.7,
    draw_labels=False,
    rm=1.1,
    title=None,
    nsolvent=36,
    d=4.0,
):
    # prepare data
    n = nsolvent + 2
    X = x.reshape(((nsolvent + 2), 2))
    # set up figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim((-d, d))
    ax.set_ylim((-d, d))
    # draw dimer
    circles = []
    circles.append(
        ax.add_patch(plt.Circle(X[0], radius=0.5 * rm, color=dimercolor, alpha=alpha))
    )
    circles.append(
        ax.add_patch(plt.Circle(X[1], radius=0.5 * rm, color=dimercolor, alpha=alpha))
    )
    # draw solvent
    for x_ in X[2:]:
        circles.append(
            ax.add_patch(
                plt.Circle(
                    x_,
                    radius=0.5 * rm,
                    color=solventcolor,
                    alpha=alpha,
                    ls="-",
                    lw=0.4,
                    fill=True,
                    ec="k",
                )
            )
        )
    ax.add_patch(plt.Rectangle((-d, -d), 2 * d, 0.5 * rm, color=boxcolor, linewidth=0))
    ax.add_patch(
        plt.Rectangle((-d, d - 0.5 * rm), 2 * d, 0.5 * rm, color=boxcolor, linewidth=0)
    )
    ax.add_patch(plt.Rectangle((-d, -d), 0.5 * rm, 2 * d, color=boxcolor, linewidth=0))
    ax.add_patch(
        plt.Rectangle((d - 0.5 * rm, -d), 0.5 * rm, 2 * d, color=boxcolor, linewidth=0)
    )
    # draw indices of particles
    if draw_labels:
        for i in range(n):
            ax.text(X[i, 0], X[i, 1], str(i))
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")
    # return(fig, ax, circles)


def draw_check(
    nn, energy, x, draw_z=False, draw_intermediates=True, verbose=0, filename=None
):
    # make sure x is numpy array
    x = np.array(x)
    n_samples = len(x)
    if len(x.shape) == 1:
        n_samples = 1
    pred = nn.model.predict(
        [
            x.reshape(n_samples, nn.dim),
            np.random.randn(n_samples, nn.dim).astype(np.float32),
        ]
    )
    pred_dict = nn.split_output(pred, return_dict=True)
    x = pred_dict["x"]
    w = pred_dict["w"]
    y_noisy = pred_dict["y_noisy"]
    sigma_x = pred_dict["sigma_x"]
    z = pred_dict["z"]
    sigma_y = pred_dict["sigma_y"]
    intermediate_outputs = pred_dict["intermediate_outputs"]
    # x, w, y, y_noisy, sigma_x, z, sigma_y, intermediate_outputs = nn.split_output(pred)
    n_plots = 2 + len(intermediate_outputs) * draw_intermediates + draw_z
    fig, axs = plt.subplots(n_samples, n_plots, figsize=(n_plots * 5, n_samples * 5))
    if n_samples == 1:
        axs = [axs]
    for k, ax in enumerate(axs):
        draw_config(
            x[k],
            ax=ax[0],
            draw_labels=True,
            title="X energy: " + str(energy(x[k].reshape(1, nn.dim))),
        )
        if draw_intermediates:
            for i, i_out in enumerate(intermediate_outputs):
                draw_config(
                    i_out[k],
                    ax=ax[i + 1],
                    draw_labels=True,
                    title="Intermediate " + str(i),
                )
        draw_config(
            y_noisy[k],
            ax=ax[len(intermediate_outputs) * draw_intermediates + 1],
            draw_labels=True,
            title="Y energy: " + str(energy(y_noisy[k].reshape(1, nn.dim))),
        )
        if draw_z:
            draw_config(z[k], ax=ax[-1], draw_labels=True, title="Z")
        if verbose > 0:
            delta_e = energy(y_noisy[k].reshape(1, nn.dim)) - energy(
                x[k].reshape(1, nn.dim)
            )
            print("delta energy:", delta_e)
            log_pprop_xy_1 = np.sum(w[k] ** 2) / 2.0
            log_pprop_xy_2 = np.sum(np.log(sigma_x[k]))
            w_y = (x[k] - z[k]) / sigma_y[k]
            log_pprop_yx_1 = np.sum(w_y ** 2) / 2.0
            log_pprop_yx_2 = np.sum(np.log(sigma_y[k]))
            reversibility = (
                -log_pprop_xy_1 - log_pprop_xy_2 + log_pprop_yx_1 + log_pprop_yx_2
            )
            print("reversibility:", reversibility)
            print("Pacc: ", 1.0 / (np.exp(delta_e + reversibility) + 1))
        if verbose > 1:
            # proposal probabilities
            print("mean sigma_x:", np.mean(sigma_x[k]))
            print("mean sigma_y:", np.mean(sigma_y[k]))
            print("log(prop_xy) 1:", log_pprop_xy_1)
            print("log(prop_xy) 2:", log_pprop_xy_2)
            print("log(prop_yx) 1:", log_pprop_yx_1)
            print("log(prop_yx) 2:", log_pprop_yx_2)
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")


def test_network(
    data,
    nn,
    energy,
    prefix="",
    generate_energy_histograms=True,
    generate_position_histograms=True,
    nsamples=2,
):
    # test the network
    # generate plot of random samples of test dataset
    if nsamples != 0:
        indcs = np.random.randint(0, len(data.traj_closed_test), size=nsamples)
        samples_open = data.traj_open_test[indcs]
        samples_closed = data.traj_closed_test[indcs]
        filename = None
        if prefix is not None:
            filename = prefix + "outputs.pdf"
        draw_check(
            nn,
            energy,
            np.concatenate([samples_open, samples_closed]),
            filename=filename,
        )

    # generate plots and histograms of test dataset
    out_open = nn.generate_output(data.traj_open_test, return_dict=True)
    x_open = out_open["x"]
    y_open = out_open["y"]
    y_noisy_open = out_open["y_noisy"]
    out_closed = nn.generate_output(data.traj_closed_test, return_dict=True)
    x_closed = out_closed["x"]
    y_closed = out_closed["y"]
    y_noisy_closed = out_closed["y_noisy"]
    E_x_open = energy(x_open)
    E_x_closed = energy(x_closed)
    E_y_open = energy(y_noisy_open)
    E_y_closed = energy(y_noisy_closed)

    # generate energy histograms
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
    axs[0].hist(
        E_x_closed,
        bins=100,
        range=(-100, 0),
        alpha=0.7,
        label="Ex   <Ex-Ey>= " + str(np.mean(E_x_closed - E_y_closed)),
    )
    axs[0].hist(
        E_y_closed,
        bins=100,
        range=(-100, 0),
        alpha=0.7,
        label="Ey   <Ey>= " + str(np.mean(E_y_closed)),
    )
    axs[0].set_title("closed input")
    axs[0].legend(loc="upper right")
    axs[1].hist(
        E_x_open,
        bins=100,
        range=(-100, 0),
        alpha=0.7,
        label="Ex   <Ex-Ey>= " + str(np.mean(E_x_open - E_y_open)),
    )
    axs[1].hist(
        E_y_open,
        bins=100,
        range=(-100, 0),
        alpha=0.7,
        label="Ey   <Ey>= " + str(np.mean(E_y_open)),
    )
    axs[1].set_title("open input")
    axs[1].legend(loc="upper right")
    if prefix is not None:
        fig.savefig(prefix + "energy_histograms.pdf", bbox_inches="tight")

    # generate dimer distance histograms
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
    ddist_x_closed = np.linalg.norm(x_closed[:, :2] - x_closed[:, 2:4], axis=1)
    ddist_y_closed = np.linalg.norm(
        y_noisy_closed[:, :2] - y_noisy_closed[:, 2:4], axis=1
    )
    ddist_x_open = np.linalg.norm(x_open[:, :2] - x_open[:, 2:4], axis=1)
    ddist_y_open = np.linalg.norm(y_noisy_open[:, :2] - y_noisy_open[:, 2:4], axis=1)
    axs[0].hist(ddist_x_closed, alpha=0.7, bins=100)
    axs[0].hist(ddist_y_open, alpha=0.7, bins=100)
    axs[0].set_title("closed state")
    axs[1].hist(ddist_x_open, alpha=0.7, bins=100)
    axs[1].hist(ddist_y_closed, alpha=0.7, bins=100)
    axs[1].set_title("open state")
    if prefix is not None:
        fig.savefig(prefix + "dimer_distance_histogram.pdf", bbox_inches="tight")

    # generate 2D histograms of solvent positions
    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 10))
    axs[0][0].hist2d(
        np.ndarray.flatten(x_open[:, 4::2]),
        np.ndarray.flatten(x_open[:, 5::2]),
        bins=100,
    )
    axs[0][1].hist2d(
        np.ndarray.flatten(y_closed[:, 4::2]),
        np.ndarray.flatten(y_closed[:, 5::2]),
        bins=100,
    )
    axs[1][0].hist2d(
        np.ndarray.flatten(x_closed[:, 4::2]),
        np.ndarray.flatten(x_closed[:, 5::2]),
        bins=100,
    )
    axs[1][1].hist2d(
        np.ndarray.flatten(y_open[:, 4::2]),
        np.ndarray.flatten(y_open[:, 5::2]),
        bins=100,
    )
    axs[0][0].set_title("open")
    axs[0][1].set_title("opening")
    axs[1][0].set_title("closed")
    axs[1][1].set_title("closing")
    axs[0][0].set_aspect("equal")
    axs[0][1].set_aspect("equal")
    axs[1][0].set_aspect("equal")
    axs[1][1].set_aspect("equal")
    if prefix is not None:
        fig.savefig(prefix + "solvent_position_histogram.pdf", bbox_inches="tight")


def map_to_reference(
    X, reference_open, reference_closed=None, dimer_split=1.5, nsolvent=36
):
    """
    Map solvent particle indices to reference configurations. If reference_closed is None, all particles will be
    mapped to the open state.
    Parameters
    ----------
    X : List of configurations to be mapped
    reference_open : Reference configuration of the open state
    reference_closed : Reference configuration of the closed state. If None, everything will be mapped to the open
    dimer_split : Where to distinguish between open and closed state
    nsolvent : Number of solvent particles

    Returns
    -------
    X_assigned : Same shape as X but with solvent labels reassigned such that the sum of distances to the reference is
                 minimized
    """
    X_assigned = np.zeros_like(X)
    for i in range(len(X)):
        X_solvents = X[i, 4:].reshape(nsolvent, 2)
        if reference_closed is not None:
            if np.abs(X[i, 0] - X[i, 2]) < dimer_split:
                reference = reference_closed
            else:
                reference = reference_open
        else:
            reference = reference_open
        reference_solvents = reference[4:].reshape(nsolvent, 2)
        C = sd.cdist(reference_solvents, X_solvents, metric="sqeuclidean")
        assignment = linear_sum_assignment(C)
        assignment = np.array(assignment)
        if X[i, 0] <= X[i, 2]:
            X_assigned[i, :4] = np.copy(X[i, :4])
        else:
            X_assigned[i, :2] = np.copy(X[i, 2:4])
            X_assigned[i, 2:4] = np.copy(X[i, :2])
        X_assigned[i, 4:] = np.copy(X_solvents[assignment].reshape(2 * nsolvent))
    return X_assigned


def reference_mapper(reference_open, reference_closed, dimer_split=1.5, nsolvent=36):
    return partial(
        map_to_reference,
        reference_open=reference_open,
        reference_closed=reference_closed,
        dimer_split=dimer_split,
        nsolvent=nsolvent,
    )


def sample_MC(
    x0, n_steps, energy_function, noise=0.03, stride=-1, verbose=False, beta=1.0
):
    n_configs = x0.shape[0]
    n_particles = x0.shape[1]
    if stride > 0:
        x_traj = np.zeros((n_steps // stride + 1, n_configs, n_particles), np.float32)
        x_traj[0, :] = x0
        x = np.array(x0)
    else:
        x = np.array(x0)
    n_accepted = 0
    for i in range(n_steps):
        # perform local steps
        y = x + noise * np.random.normal(size=(n_configs, n_particles))
        dE = energy_function(y) - energy_function(x)
        p_acc = np.exp(-beta * dE)

        # accept or reject and append to data array
        rands = np.random.rand(n_configs)
        accepted = np.where(rands < p_acc)[0]
        rejected = np.where(rands >= p_acc)[0]
        if verbose:
            n_accepted += len(accepted)
        x[accepted] = y[accepted]
        x[rejected] = x[rejected]
        if stride > 0 and not i % stride:
            x_traj[i // stride + 1] = x
    if stride < 0:
        x_traj = x
    if verbose:
        print(
            "Ratio of accepted moves: {}".format(
                float(n_accepted) / float(n_steps * n_configs)
            )
        )
    return x_traj


def init_positions(dimer_distance=1.0, n=36, d=4):
    """ Initializes particles positions in a box

    Parameters:
    -----------
    n : int
        number of solvent particles, must be a square of an int.
    d : float
        box dimensions [-d...d] x [-d...d]

    """
    # dimer
    pos = []
    pos.append(np.array([-0.5 * dimer_distance, 0]))
    pos.append(np.array([0.5 * dimer_distance, 0]))
    # solvent particles
    sqrtn = int(np.sqrt(n))
    locs = np.linspace(-d, d, sqrtn + 2)[1:-1]
    for i in range(0, sqrtn):
        for j in range(0, sqrtn):
            pos.append(np.array([locs[i], locs[j]]))
    return np.array(pos).reshape((1, 2 * (n + 2)))
