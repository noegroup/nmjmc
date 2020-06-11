import numpy as np
import uuid
import os

import tensorflow as tf
import pickle
from tensorflow import keras
import nmjmc.systems as systems
from .revbase import ReversibleNetwork

Model = keras.models.Model
Add = keras.layers.Add
Multiply = keras.layers.Multiply
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Input = keras.layers.Input
Lambda = keras.layers.Lambda
concatenate = keras.layers.concatenate
adam = keras.optimizers.Adam
max_norm = keras.constraints.max_norm

Subtract = Lambda(
    lambda inputs: inputs[0] - inputs[1]
)  # , output_shape=lambda shapes: shapes[0])


class NeuralMJMCNetwork(ReversibleNetwork):
    """
    This class implements the RNVP network as described in Dinh et al., "Density estimation using real NVP" (2017)
    """

    def __init__(
        self,
        s_nodes,
        t_nodes,
        block_length,
        scaling_mu=0.001,
        scaling_s=0.001,
        dim=76,
        dimer_split=1.5,
        layer_activation="relu",
        system=systems.Particles(),
        split_cond=None,
        split_dims_conditions=None,
    ):
        super().__init__(
            0,
            block_length,
            scaling_mu,
            1.0,
            dim,
            False,
            dimer_split,
            layer_activation,
            system,
            split_cond,
        )
        # Assemble network
        self.s_nodes = s_nodes  # dimensions of the networks in the NICER layers
        self.t_nodes = t_nodes
        self.scaling_s = scaling_s
        # number of NICER blocks
        self.dimer_split = dimer_split
        self.system = system
        self.N = self.dim // 2
        # generate random ID for temporary saving
        self.uid = uuid.uuid4().hex[:6].upper()
        self.model = None
        self.generate_network()
        self.split_dims_conditions = split_dims_conditions

    def generate_layers(self):
        # define RNVP blocks
        self.RNVP_blocks = []
        for i in range(self.nblocks):
            self.RNVP_blocks.append(
                self.make_layers(
                    self.s_nodes,
                    self.t_nodes,
                    None,
                    None,
                    name_layers="block_" + str(i),
                    activation=self.layer_activation,
                    term_linear=True,
                )
            )
            if np.in1d(i, self.intermediate_output_indices):
                self.RNVP_blocks.append("output_" + str(i))

    def generate_network(self):

        input_x = Input(shape=(self.dim,), name="x")

        self.generate_layers()

        # compute the forward and backward state from the input
        y_forward, j_y_forward = self.assemble_forward(input_x, [], self.RNVP_blocks)
        y_backward, j_y_backward = self.assemble_backward(input_x, [], self.RNVP_blocks)

        # decide if state was open in first place and keep the respective y only
        x_yf_yb = concatenate([input_x, y_forward, y_backward])
        y = Lambda(
            self.select_direction,
            arguments={"size": self.dim, "dimer_split": self.dimer_split},
        )(x_yf_yb)
        x_jf_jb = concatenate([input_x, j_y_forward, j_y_backward])
        j_x = Lambda(
            self.select_direction,
            arguments={"size": self.N, "dimer_split": self.dimer_split},
        )(x_jf_jb)

        outputs = [input_x, y, j_x]
        outputs = concatenate(outputs, name="concatenate_outputs")

        # define model
        self.model = Model(inputs=input_x, outputs=outputs)
        print(
            "Model created successfully. Number of parameters:",
            self.model.count_params(),
        )

    def add_intermediate_output(self, x0, x1, j, outputs):
        """
        Add current output to outputsj_x,

        Parameters:
        ----------
        outputs : list
                List of intermediate outputs
        """
        x0x1 = concatenate([x0, x1, j])
        outputs.append(
            Lambda(self.merge_x0x1j, name="intermediate_output" + str(len(outputs)))(
                x0x1
            )
        )
        return x0, x1

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
        pred = self.model.predict(x)
        if split_output:
            return self.split_output(pred, return_dict=return_dict)
        else:
            return pred

    def merge_x0x1j(self, x0x1):
        x0 = x0x1[:, 0 : int(self.dim / 2)]
        x1 = x0x1[:, int(self.dim / 2) : self.dim]
        j = x0x1[:, self.dim :]
        j = tf.reshape(j, [-1, self.N])
        x0_exp = tf.expand_dims(x0, 2)
        x1_exp = tf.expand_dims(x1, 2)
        concat_x0x1 = tf.concat([x0_exp, x1_exp], 2)
        concat_x0x1 = tf.reshape(concat_x0x1, [-1, self.dim])
        return tf.concat([concat_x0x1, j], 1)

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
        j = []
        j_intermediate = []
        for i, block in enumerate(nicer_blocks):
            if isinstance(block, list):
                x0, x1, _j = self.add_forward(x0, x1, block)
                j.append(_j)
                j_intermediate.append(_j)
            elif not skip_intermediates:
                x0, x1 = self.add_intermediate_output(
                    x0, x1, Add()(j_intermediate), outputs
                )
                j_intermediate = []
        y0y1_forward = concatenate([x0, x1])
        y_forward = Lambda(self.merge_x0x1)(y0y1_forward)
        log_j = Add()(j)
        return y_forward, log_j

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
        j_intermediate = []
        for i in reversed(range(len(nicer_blocks))):
            block = nicer_blocks[i]
            if isinstance(block, list):
                x0, x1, _j = self.add_backward(x0, x1, block)
                j.append(_j)
                j_intermediate.append(_j)
            elif not skip_intermediates:
                x0, x1 = self.add_intermediate_output(
                    x0, x1, Add()(j_intermediate), outputs
                )
                j_intermediate = []
        y0y1_backward = concatenate([x0, x1])
        y_backward = Lambda(self.merge_x0x1)(y0y1_backward)
        log_j = Add()(j)
        return y_backward, log_j

    def add_forward(self, x0, x1, block):
        """
        Assembles a RNVP block in the forward sense as created by make_layers

        Parameters:
        ----------
        x0 : tensor
        x1 : tensor
        rnvp_block : list as created by make_layers
        Returns:
        -------
        z0, z1 : tensors
                outputs in the RNVP layer sense
        """
        f_s_layers, f_t_layers = block[0]
        g_s_layers, g_t_layers = block[1]
        s0 = self.concat_scaling_layers(x0, f_s_layers)
        s0_exp = Lambda(tf.exp)(s0)
        t0 = self.concat_layers(x0, f_t_layers)
        y0 = x0
        y1 = Multiply()([x1, s0_exp])
        y1 = Add()([y1, t0])
        s1 = self.concat_scaling_layers(y1, g_s_layers)
        s1_exp = Lambda(tf.exp)(s1)
        t1 = self.concat_layers(y1, g_t_layers)
        z0 = Multiply()([y0, s1_exp])
        z0 = Add()([z0, t1])
        z1 = y1
        return z0, z1, Add()([s0, s1])

    def add_backward(self, z0, z1, block):
        """
        Assembles a RNVP block in the backward sense as created by make_layers

        Parameters:
        ----------
        z0 : tensor
        z1 : tensor
        block : list as created by make_layers
        output : boolean
                If true add intermediate result to intermediate_outputs
        intermediate_outputs : list
                Intermediate outputs are appended to this list
        Returns:
        -------
        z0, z1 : tensors
                outputs in the RNVP layer sense
        """
        f_s_layers, f_t_layers = block[0]
        g_s_layers, g_t_layers = block[1]
        t1 = self.concat_layers(z1, g_t_layers)
        s1 = Lambda(tf.negative)(self.concat_scaling_layers(z1, g_s_layers))
        s1_exp = Lambda(tf.exp)(s1)
        y1 = z1
        y0 = Subtract([z0, t1])
        y0 = Multiply()([y0, s1_exp])
        t0 = self.concat_layers(y0, f_t_layers)
        s0 = Lambda(tf.negative)(self.concat_scaling_layers(y0, f_s_layers))
        s0_exp = Lambda(tf.exp)(s0)
        x0 = y0
        x1 = Subtract([y1, t0])
        x1 = Multiply()([x1, s0_exp])
        return x0, x1, Add()([s0, s1])

    def make_layers(
        self,
        s_nodes,
        t_nodes,
        s_dropout,
        t_dropout,
        name_layers,
        activation="relu",
        term_linear=True,
    ):
        """
        Make a RNVRP block. This is not assembled yet.
        Parameters:
        ----------
        s_nodes : list of integers
                nodes in each layer of the scaling network
        t_nodes : list of integers
                nodes in each layer of the translation network
        s_dropout : list of float
                dropout ratio after each dense layer in the scaling network
        t_dropout : list of float
                dropout ratio after each dense layer in the translation network
        name_layers : string
                name of the layers in this block
        activation : string
               name of activation function
        term_linear : boolean
                add a linear layer at the end of the network
        Returns:
        -------
        RNVP_layers : list
                list of RNVP layers to be assembled by the forward or backward function
        """
        f_s_layers = []
        f_t_layers = []
        g_s_layers = []
        g_t_layers = []
        for i in range(len(s_nodes)):
            f_s_layers.append(
                Dense(
                    s_nodes[i],
                    activation=activation,
                    name=name_layers + "_f_s_" + str(i),
                    kernel_constraint=max_norm(self.kernel_max_norm),
                )
            )
            if s_dropout is not None:
                f_s_layers.append(Dropout(s_dropout[i], noise_shape=None, seed=None))
            g_s_layers.append(
                Dense(
                    s_nodes[i],
                    activation=activation,
                    name=name_layers + "_g_s_" + str(i),
                    kernel_constraint=max_norm(self.kernel_max_norm),
                )
            )
            if s_dropout is not None:
                g_s_layers.append(Dropout(s_dropout[i], noise_shape=None, seed=None))
        f_s_layers.append(
            Dense(int(self.dim / 2), activation="tanh", name=name_layers + "_f_s_tanh")
        )
        g_s_layers.append(
            Dense(int(self.dim / 2), activation="tanh", name=name_layers + "_g_s_tanh")
        )
        f_s_layers.append(
            Dense(1, activation="linear", name=name_layers + "_f_s_linear")
        )
        g_s_layers.append(
            Dense(1, activation="linear", name=name_layers + "_g_s_linear")
        )
        for i in range(len(t_nodes)):
            f_t_layers.append(
                Dense(
                    t_nodes[i],
                    activation=activation,
                    name=name_layers + "_f_t_" + str(i),
                    kernel_constraint=max_norm(self.kernel_max_norm),
                )
            )
            if t_dropout is not None:
                f_t_layers.append(Dropout(t_dropout[i], noise_shape=None, seed=None))
            g_t_layers.append(
                Dense(
                    t_nodes[i],
                    activation=activation,
                    name=name_layers + "_g_t_" + str(i),
                    kernel_constraint=max_norm(self.kernel_max_norm),
                )
            )
            if t_dropout is not None:
                g_t_layers.append(Dropout(t_dropout[i], noise_shape=None, seed=None))
        if term_linear:
            f_t_layers.append(
                Dense(
                    int(self.dim / 2),
                    activation="linear",
                    name=name_layers + "_f_t_linear_layer",
                )
            )
            g_t_layers.append(
                Dense(
                    int(self.dim / 2),
                    activation="linear",
                    name=name_layers + "_g_t_linear_layer",
                )
            )
        f_t_layers.append(
            Lambda(
                self.scale,
                arguments={"factor": self.scaling_mu},
                name=name_layers + "_f_t_scale_mu",
            )
        )
        g_t_layers.append(
            Lambda(
                self.scale,
                arguments={"factor": self.scaling_mu},
                name=name_layers + "_g_t_scale_mu",
            )
        )
        f_layers = [f_s_layers, f_t_layers]
        g_layers = [g_s_layers, g_t_layers]
        return [f_layers, g_layers]

    def concat_scaling_layers(self, inp, layers):
        """
        Serially concatenates a list of layers after the input, and returns a link to the last layer. The last layer
        in layers scales the second to last layer in layers.
        """
        layer = inp
        for l in layers[:-1]:
            layer = l(layer)
        scaling = layers[-1](inp)
        scaling = Lambda(self.scale, arguments={"factor": self.scaling_s})(scaling)
        return Multiply()([layer, scaling])

    def save_network(self, filename):
        network_params = {
            "s_nodes": self.s_nodes,
            "t_nodes": self.t_nodes,
            "block_length": self.block_length,
            "scaling_mu": self.scaling_mu,
            "scaling_s": self.scaling_s,
            "dim": self.dim,
            "dimer_split": self.dimer_split,
            "layer_activation": self.layer_activation,
            "split_cond": self.split_cond,
            "system": self.system,
        }
        pickle.dump(network_params, open(filename + ".mnn", "wb"))
        self.save_weights(filename + "_weights.h5")

    @classmethod
    def load_network(cls, filename):
        np = pickle.load(open(filename + ".mnn", "rb"))
        # this is just for backward compatibility
        nn = cls(
            np["s_nodes"],
            np["t_nodes"],
            np["block_length"],
            np["scaling_mu"],
            np["scaling_s"],
            np["dim"],
            np["dimer_split"],
            np["layer_activation"],
            np["system"],
            np["split_cond"],
        )
        nn.load_weights(filename + "_weights.h5")
        return nn

    def trainable_rnvp(self, trainable=True):
        for block in self.RNVP_blocks:
            if not isinstance(block, str):
                for network in block:
                    for layer in network:
                        if hasattr(layer, "trainable"):
                            layer.trainable = trainable

    def split_output(self, out, return_dict=False):
        """
        Template function for splitting the output. This is instantiated later on in the code

        Returns:
        -------
        List containing the outputs
        """
        x = out[:, : self.dim]
        y = out[:, self.dim : 2 * self.dim]
        j_x = out[:, 2 * self.dim : 2 * self.dim + self.dim // 2]

        if return_dict:
            dict = {"x": x, "y": y, "j_x": j_x}
            return dict
        else:
            return [x, y, j_x]

    def train_pair(
        self,
        x0,
        y0,
        loss,
        validation_data=None,
        learning_rate=0.001,
        batchsize=2000,
        nepochs=100,
        callbacks=[],
        clipnorm=None,
        shuffle=True,
    ):
        if clipnorm is not None:
            optimizer = adam(lr=learning_rate, clipnorm=clipnorm)
        else:
            optimizer = adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss)
        # tensorboard = TensorBoard(log_dir="./logs/")
        N_data = len(x0)
        loss = self.model.fit(
            x0,
            y0,
            epochs=nepochs,
            batch_size=batchsize,
            validation_split=0.0,
            shuffle=shuffle,
            callbacks=callbacks,
            validation_data=validation_data,
        )
