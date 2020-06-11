import numpy as np
import uuid

import tensorflow as tf
import tensorflow.keras as keras
import nmjmc.systems as systems

Model = keras.models.Model
Add = keras.layers.Add
Multiply = keras.layers.Multiply
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Input = keras.layers.Input
Lambda = keras.layers.Lambda
concatenate = keras.layers.concatenate
adam = keras.optimizers.Adam
max_norm = keras.constraints.max_norm
"""
These can be used with newer version of tensorflow. However, we want to keep compatibility with tf 1.4
from keras.models import Model
from keras.layers import Add, Multiply, Dense, Dropout, Input, Lambda, concatenate, BatchNormalization
from keras.optimizers import Adam as adam
from keras.constraints import max_norm
from keras import regularizers
"""


Subtract = Lambda(
    lambda inputs: inputs[0] - inputs[1]
)  # , output_shape=lambda shapes: shapes[0])


class ReversibleNetwork:
    def __init__(
        self,
        nintermediates,
        block_length,
        scaling_mu=0.001,
        scaling_sigma=0.005,
        dim=76,
        fixed_sigma=False,
        dimer_split=1.5,
        layer_activation="relu",
        system=systems.Particles(),
        split_cond=None,
    ):
        # Assemble network
        self.dim = dim
        self.scaling_mu = scaling_mu
        self.scaling_sigma = scaling_sigma
        self.block_length = block_length
        self.nintermediates = nintermediates
        self.nblocks, self.intermediate_output_indices = self.block_parameters(
            self.nintermediates, self.block_length
        )
        # number of NICER blocks
        self.kernel_max_norm = 3.0
        self.fixed_sigma = fixed_sigma
        self.dimer_split = dimer_split
        self.layer_activation = layer_activation
        self.system = system
        # generate random ID for temporary saving
        self.uid = uuid.uuid4().hex[:6].upper()
        self.model = None
        if split_cond is None:
            self.split_cond = self._split_cond
        else:
            self.split_cond = split_cond

    @staticmethod
    def block_parameters(noutputs, nintermediate_blocks):
        """
        Generates parameters for nblocks and intemerdiate_output_indices in case the number of blocks in between each
        output is constant.
        Parameters:
        ----------
        noutputs : int
                number of intermediate outputs
        nintermediate_blocks : int
                number of blocks between each output
        """
        return (
            (noutputs + 1) * nintermediate_blocks,
            [nintermediate_blocks * i - 1 for i in range(1, noutputs + 1)],
        )

    @staticmethod
    def scale(x, factor):
        return factor * x

    def _fixed_sigma(self, x):
        return x * 0.0 + self.fixed_sigma

    def select_direction(self, x_f_b, size, dimer_split):
        """
        Select either the forward or backward direction of the NICER network depending on the
        distance of the dimer.
        Parameters:
        ----------
        x_f_b : 2D tensor
            Tensor of concatenated input, forward output and backward output
        size : int
            Size of the respective output
        dimer_split : float
            Point at which open or closed state are distinguished
        """
        x = x_f_b[:, : self.dim]
        f = x_f_b[:, self.dim : (self.dim + size)]
        b = x_f_b[:, (self.dim + size) : (self.dim + 2 * size)]
        cond = self.split_cond(x, dimer_split)
        # cond = tf.reshape(cond, [-1, 1])
        return tf.where(cond, f, b)

    def _split_cond(self, x, dimer_split):
        return tf.abs(x[:, 0] - x[:, 2]) < dimer_split

    def add_intermediate_output(self, x0, x1, outputs):
        """
        Add current output to outputs

        Parameters:
        ----------
        outputs : list
                List of intermediate outputs
        """
        x0x1 = concatenate([x0, x1])
        outputs.append(
            Lambda(self.merge_x0x1, name="intermediate_output" + str(len(outputs)))(
                x0x1
            )
        )
        return x0, x1

    def assemble_forward(self, x, outputs, nicer_blocks, skip_intermediates=False):
        """
        Assemble nicer blocks in the forward sense. If there is an intermediate output,
        it is added to outputs

        Parameters:
        ----------
        x : 2D tensor
            input configuration
        outputs : list
            intermediate outputs are added to this list
        nicer_blocks : list
            nicer blocks with intermediate outputs
        skip_intermediates :

        Returns:
        -------
        y_forward : 2D tensor
            transformed configuration in the forward sense
        """
        x0 = Lambda(self.split_x0)(x)
        x1 = Lambda(self.split_x1)(x)
        for i, block in enumerate(nicer_blocks):
            if isinstance(block, list):
                x0, x1 = self.add_forward(x0, x1, block)
            elif not skip_intermediates:
                x0, x1 = self.add_intermediate_output(x0, x1, outputs)
        y0y1_forward = concatenate([x0, x1])
        y_forward = Lambda(self.merge_x0x1)(y0y1_forward)
        return y_forward

    def assemble_backward(self, x, outputs, nicer_blocks, skip_intermediates=False):
        """
        Assemble nicer blocks in the backward sense. If there is an intermediate output,
        it is added to outputs

        Parameters:
        ----------
        x : 2D tensor
            input configuration
        outputs : list
            intermediate outputs are added to this list
        nicer_blocks : list of nicer blocks with intermediate outputs

        Returns:
        -------
        y_backward : 2D tensor
            transformed configuration in the backward sense
        """
        x0 = Lambda(self.split_x0)(x)
        x1 = Lambda(self.split_x1)(x)
        j = []
        for i in reversed(range(len(nicer_blocks))):
            block = nicer_blocks[i]
            if isinstance(block, list):
                x0, x1, _j = self.add_backward(x0, x1, block)
            elif not skip_intermediates:
                x0, x1 = self.add_intermediate_output(x0, x1, outputs)
        y0y1_backward = concatenate([x0, x1])
        y_backward = Lambda(self.merge_x0x1)(y0y1_backward)
        return y_backward

    @staticmethod
    def concat_layers(inp, layers):
        """ Serially concatenates a list of layers after the input, and returns a link to the last layer
        """
        layer = inp
        for l in layers:
            layer = l(layer)
        return layer

    @staticmethod
    def split_x0(x):
        return x[:, ::2]

    @staticmethod
    def split_x1(x):
        return x[:, 1::2]

    def merge_x0x1(self, x0x1):
        x0 = x0x1[:, 0 : int(self.dim / 2)]
        x1 = x0x1[:, int(self.dim / 2) : self.dim]
        x0_exp = tf.expand_dims(x0, 2)
        x1_exp = tf.expand_dims(x1, 2)
        concat_x0x1 = tf.concat([x0_exp, x1_exp], 2)
        return tf.reshape(concat_x0x1, [-1, self.dim])

    def split_output(self, out, return_dict=False):
        """
        Template function for splitting the output. This is instantiated later on in the code

        Returns:
        -------
        List containing the outputs
        """
        x = out[:, : self.dim]
        w = out[:, self.dim : 2 * self.dim]
        y = out[:, 2 * self.dim : 3 * self.dim]
        y_noisy = out[:, 3 * self.dim : 4 * self.dim]
        sigma_x = out[:, 4 * self.dim : 5 * self.dim]
        z = out[:, 5 * self.dim : 6 * self.dim]
        sigma_y = out[:, 6 * self.dim : 7 * self.dim]
        intermediate_ys = out[
            :, 7 * self.dim : (7 * self.dim + self.nintermediates * self.dim)
        ]
        intermediate_return = [
            intermediate_ys[:, i * self.dim : (i + 1) * self.dim]
            for i in range(self.nintermediates)
        ]
        if return_dict:
            return {
                "x": x,
                "w": w,
                "y": y,
                "y_noisy": y_noisy,
                "sigma_x": sigma_x,
                "z": z,
                "sigma_y": sigma_y,
                "intermediate_outputs": intermediate_return,
            }
        else:
            return [x, w, y, y_noisy, sigma_x, z, sigma_y, intermediate_return]

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def train_pair(
        self,
        x0,
        y0,
        loss,
        validation_data=None,
        energy=None,
        energy_cap=0.0,
        learning_rate=0.001,
        batchsize=2000,
        redraws=10,
        nepochs=100,
        return_samples=False,
        noise_scale=1.0,
        verbose=True,
        callbacks=[],
        clipnorm=None,
        ncopies=1,
        shuffle=True,
    ):
        if clipnorm is not None:
            optimizer = adam(lr=learning_rate, clipnorm=clipnorm)
        else:
            optimizer = adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss)
        # tensorboard = TensorBoard(log_dir="./logs/")
        N_data = len(x0)
        x0 = np.concatenate([x0 for i in range(ncopies)])
        y0 = np.concatenate([y0 for i in range(ncopies)])
        for i in range(redraws):
            print("Redraw ", i, "/", redraws, ":")
            w = noise_scale * np.random.normal(
                size=(N_data * ncopies, self.dim)
            ).astype(np.float32)
            if energy_cap != 0:
                prediction = self.model.predict([x0, w])
                (
                    x,
                    w,
                    y,
                    y_noisy,
                    sigma_x,
                    z,
                    sigma_y,
                    intermediate_y,
                ) = self.split_output(prediction)
                ey = energy(y)
                ey2 = energy(z)
                mild_energy = np.where(
                    np.logical_and(ey < energy_cap, ey2 < energy_cap)
                )
                xTrain = x0[mild_energy[0]]
                yTrain = y0[mild_energy[0]]
                w = w[mild_energy[0]]
                print(len(mild_energy[0]))
            else:
                xTrain = x0
                yTrain = y0
            if not (validation_data is None):
                w_validation = np.random.normal(
                    size=(validation_data[0].shape[0], self.dim)
                )
                validation = [[validation_data[0], w_validation], validation_data[1]]
            else:
                validation = None
            loss = self.model.fit(
                [xTrain, w],
                yTrain,
                epochs=nepochs,
                batch_size=batchsize,
                validation_split=0.0,
                shuffle=shuffle,
                callbacks=callbacks,
                validation_data=validation,
            )

    def train_bootstrap(
        self,
        x0,
        size,
        loss,
        nepochs=100,
        ndraws=10,
        batchsize=4096,
        max_training_data=500000,
        local_ratio=0.3,
        learning_rate=0.001,
        reassign_labels=None,
        callbacks=[],
    ):
        optimizer = adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss)
        training_set = []
        training_set_cp = []
        for i in range(ndraws):
            print("draw: ", i)
            if i > 0:
                training_set_cp = training_set
            training_set = self.generate_combined_batch(
                x0, size, local_ratio=local_ratio, reassign_labels=reassign_labels
            )
            x0 = np.copy(training_set[-1, :, :])
            training_set = np.concatenate(training_set)
            if i > 0:
                if len(training_set_cp) >= max_training_data:
                    rand_idcs = np.random.choice(
                        len(training_set_cp), len(training_set), replace=False
                    )
                    training_set_cp[rand_idcs] = training_set
                    training_set = training_set_cp
                else:
                    training_set = np.concatenate([training_set, training_set_cp])
            w = np.random.normal(size=(len(training_set), self.dim)).astype(np.float32)
            y = np.zeros_like(w)
            loss = self.model.fit(
                [training_set, w],
                y,
                epochs=nepochs,
                batch_size=batchsize,
                validation_split=0.0,
                shuffle=True,
                callbacks=callbacks,
            )

    def generate_output(
        self, x, label_assigner=None, split_output=True, return_dict=True
    ):
        if not isinstance(x, np.ndarray):
            # convert to np array
            x = np.array(x)
        x_shape = x.shape
        if len(x_shape) == 1 and x_shape[0] == self.dim:
            n_samples = 1
            x = x.reshape(n_samples, self.dim)
        else:
            n_samples = x_shape[0]
        if label_assigner is not None:
            x = label_assigner.assign_labels(x)
        pred = self.model.predict(
            [x, np.random.randn(n_samples, self.dim).astype(np.float32)]
        )
        if split_output:
            return self.split_output(pred, return_dict=return_dict)
        else:
            return pred

    def pacc_barker_np(self, x, w=None, y=None, sigma_x=None, z=None, sigma_y=None):
        # compute proposal probabilities
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
        log_pprop_xy = -np.sum(w ** 2, axis=1) / 2.0 - np.sum(np.log(sigma_x), axis=1)
        w_y = (x - z) / sigma_y
        log_pprop_yx = -np.sum(w_y ** 2, axis=1) / 2.0 - np.sum(np.log(sigma_y), axis=1)
        # compute energy difference
        dE = self.system.energy(y) - self.system.energy(x) + log_pprop_xy - log_pprop_yx
        pacc = 1.0 / (np.exp(dE) + 1)
        return pacc

    def pacc_mh_np(self, x, beta=1, w=None, y=None, sigma_x=None, z=None, sigma_y=None):
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
        dE = self.system.energy(y) - self.system.energy(x) + log_pprop_xy - log_pprop_yx
        pacc = np.minimum(np.exp(-dE * beta), 1.0)
        return pacc

    def pacc_mjmc(self, x, beta=1, w=None, y=None, sigma_x=None, z=None, sigma_y=None):
        # got input x as dictionary
        if isinstance(x, dict):
            pred = x
            x, y, = pred["x"], pred["y"]
        # compute proposal probabilities
        # compute energy difference
        dE = self.system.energy(y) - self.system.energy(x)
        pacc = np.minimum(np.exp(-dE * beta), 1.0)
        return pacc

    def generate_combined_batch(
        self,
        x0,
        size,
        noise=0.03,
        local_ratio=0.0,
        reassign_labels=None,
        thresholds=None,
    ):
        """
        Parameters:
        ----------
        x0 : list
            list of initial configuration from which the model generates data
        size : int
            size of batch
        noise : float
            std of the local steps
        local_ratio : float
            ratio of local i.e. small Gaussian steps
        reassign_labels : function
            function to reassign labels as to make the network use them properly.
            Can be None if no reassignment should be performed.
        Returns:
        -------
        x : numpy array
            3 dimensional array which contains Markov chain for every input configuration of length size.
            Dimensions of output are (size, number of configurations, number of particles)

        """
        print("modified")
        # initialize arrays for output
        n_configs = x0.shape[0]
        n_particles = x0.shape[1]
        x = np.zeros((size, n_configs, n_particles), np.float32)
        w = np.random.normal(size=(size, n_configs, n_particles)).astype(np.float32)
        sigma = noise * np.ones((size, x0.shape[0]), dtype=np.float32)
        y = np.zeros((n_configs, n_particles), dtype=np.float32)
        p_acc = np.zeros((n_configs), dtype=np.float32)
        mean_p_acc = np.zeros((size, 2), dtype=np.float32)
        x[0, :] = x0
        for i in range(size - 1):
            # randomly draw whether to perform global or local step
            rands = np.random.rand(n_configs)
            local_idcs = np.where(rands < local_ratio)[0]
            global_idcs = np.where(rands >= local_ratio)[0]

            # perform local steps
            # check first if list is empty
            if local_idcs.size > 0:
                y[local_idcs] = x[i, local_idcs] + noise * w[i, local_idcs]
                dE = self.system.energy(y[local_idcs]) - self.system.energy(
                    x[i, local_idcs]
                )
                # p_acc[local_idcs] = 1. / (1. + np.exp(dE))
                p_acc[local_idcs] = np.minimum(np.exp(-dE), 1.0)
                mean_p_acc[i, 1] = np.mean(p_acc[local_idcs])
                # y = relabel_solvent(y.reshape(1,dim))

            # perform global steps
            # check first if list is empty to avoid overhead
            if global_idcs.size > 0:
                if thresholds is not None:
                    # Check which of the selected configurations are within the defined volume
                    within_threshold = self.within_threshold(
                        x[i, global_idcs], thresholds
                    )
                    global_idcs_inside = global_idcs[within_threshold]
                    global_idcs_outside = global_idcs[~within_threshold]
                    # Perform local steps for configurations that are outside
                    y[global_idcs_outside] = x[
                        i, global_idcs_outside
                    ] + noise * np.random.randn(len(global_idcs_outside), self.dim)
                    dE = self.system.energy(
                        y[global_idcs_outside]
                    ) - self.system.energy(x[i, global_idcs_outside])
                    p_acc[global_idcs_outside] = np.minimum(np.exp(-dE), 1.0)
                    # Propose global steps for the ones that are inside
                    if len(global_idcs_inside) > 0:
                        pred = self.generate_output(
                            x[i, global_idcs_inside], return_dict=True
                        )
                        # Check which steps would lead to jumps outside the volume
                        y_within_threshold = self.within_threshold(
                            pred["y"], thresholds
                        )
                        idcs_inside = np.where(y_within_threshold)[0]
                        global_idcs_outside = global_idcs_inside[~y_within_threshold]
                        global_idcs_inside = global_idcs_inside[y_within_threshold]
                        if len(global_idcs_inside) > 0:
                            p_acc[global_idcs_inside] = self.pacc_mh_np(pred)[
                                idcs_inside
                            ]
                            y[global_idcs_inside] = pred["y_noisy"][idcs_inside]
                            mean_p_acc[i, 0] = np.mean(p_acc[global_idcs_inside])
                        # Propose local step for configurations that would've left the volume
                        y[global_idcs_outside] = x[
                            i, global_idcs_outside
                        ] + noise * np.random.randn(len(global_idcs_outside), self.dim)
                        dE = self.system.energy(
                            y[global_idcs_outside]
                        ) - self.system.energy(x[i, global_idcs_outside])
                        p_acc[global_idcs_outside] = np.minimum(np.exp(-dE), 1.0)
                else:
                    pred = self.generate_output(x[i, global_idcs], return_dict=True)
                    p_acc[global_idcs] = self.pacc_mh_np(pred)
                    mean_p_acc[i, 0] = np.mean(p_acc[global_idcs])
                    y[global_idcs] = pred["y_noisy"]

            # accept or reject and append to data array
            rands = np.random.rand(n_configs)
            accepted = np.where(rands < p_acc)[0]
            # print('accepted:', len(accepted))
            rejected = np.where(rands >= p_acc)[0]
            if reassign_labels is not None:
                y_accepted = reassign_labels(y[accepted])
            else:
                y_accepted = y[accepted]
            x[i + 1, accepted] = y_accepted
            x[i + 1, rejected] = x[i, rejected]
        return x, mean_p_acc

    def within_threshold(self, x, thresholds):
        """
        Check if configuration x is within the thresholds of kernel number kernel_id.
        These are defined as elliptic volumes.
        Thresholds should  be of shape (open/closed), (m_x, axis_x, m_y, axis_y)
        """
        return np.where(
            np.linalg.norm(x[:, :2] - x[:, 2:4], axis=1) > self.dimer_split,
            np.product(
                (x[:, ::2] - thresholds[0, 0]) ** 2 / thresholds[0, 1]
                + (x[:, 1::2] - thresholds[0, 2]) ** 2 / thresholds[0, 3]
                <= 1.0,
                axis=1,
            ),
            np.product(
                (x[:, ::2] - thresholds[1, 0]) ** 2 / thresholds[1, 1]
                + (x[:, 1::2] - thresholds[1, 2]) ** 2 / thresholds[1, 3]
                <= 1.0
            ),
        )
