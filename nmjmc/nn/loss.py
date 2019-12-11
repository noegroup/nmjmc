import tensorflow as tf
from functools import partial, update_wrapper


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def linlogcut(x, a=0, b=1000):
    # cutoff x after b - this should also cutoff infinities
    x = tf.where(x < b, x, b * tf.ones(tf.shape(x)))
    # log after a
    y = a + tf.where(x < a, x - a, tf.math.log(x - a + 1))
    # make sure everything is finite
    y = tf.where(tf.math.is_finite(y), y, b * tf.ones(tf.shape(y)))
    return y


class BiasedPotentialLoss:
    def __init__(self, system, bias, rnvp_network, high_energy=1e5, max_energy=1e9):
        self.system = system
        self.bias = bias
        self.nn = rnvp_network
        self.high_energy = high_energy
        self.max_energy = max_energy

    def _supervised_acceptance(
        self, y_true, y_pred, energy_function, factor_distance, factor_acceptance
    ):
        x, y, j_x = self.nn.split_output(y_pred)

        red_diff = tf.reduce_sum((y - y_true) ** 2, axis=1)
        dE = energy_function(y) - energy_function(x)

        red_jacobian = tf.reduce_sum(j_x, axis=1)

        dF = tf.reduce_mean(tf.abs(-dE + red_jacobian))

        return tf.reduce_mean(
            factor_distance * red_diff + factor_acceptance * dF, axis=0
        )

    def _unsupervised_acceptance(
        self, _, y_pred, energy_function, bias_function, factor_bias, factor_temperature
    ):
        x, y, j_x = self.nn.split_output(y_pred)

        dE = energy_function(y) - energy_function(x)
        dE_bias = bias_function(y) - bias_function(x)

        red_jacobian = tf.reduce_sum(j_x, axis=1)

        dF = (
            factor_bias * tf.abs(dE_bias)
            + (dE - factor_temperature * red_jacobian) ** 2
        )

        dF = linlogcut(dF, self.high_energy, self.max_energy)

        return tf.reduce_mean(dF)

    def initialization_loss(self, energy_function, factor_distance, factor_acceptance):
        return wrapped_partial(
            self._supervised_acceptance,
            energy_function=energy_function,
            factor_distance=factor_distance,
            factor_acceptance=factor_acceptance,
        )

    def unsupervised_acceptance(
        self, energy_function, bias_function, factor_bias, factor_temperature
    ):
        return wrapped_partial(
            self._unsupervised_acceptance,
            energy_function=energy_function,
            bias_function=bias_function,
            factor_bias=factor_bias,
            factor_temperature=factor_temperature,
        )


class NeuralMJMCLoss:
    def __init__(self, system, rnvp_network):
        self.system = system
        self.nn = rnvp_network

    def __supervised_acceptance(
        self, y_true, y_pred, energy_function, factor_supervision, factor_entropy
    ):

        x, y, z, j_x, j_y = self.nn.split_output(y_pred)

        red_diff = tf.reduce_mean(tf.reduce_sum((y - y_true) ** 2, axis=1))
        dE = tf.reduce_mean(energy_function(y) - energy_function(x)) ** 2

        red_jacobian = tf.reduce_mean((tf.reduce_sum(j_x, axis=1))) ** 2

        return factor_supervision * red_diff + factor_entropy * red_jacobian + dE

    def _supervised_acceptance(
        self, y_true, y_pred, energy_function, factor_supervision, factor_entropy
    ):

        x, y, z, j_x, j_y = self.nn.split_output(y_pred)

        red_diff = tf.reduce_mean(tf.reduce_sum((y - y_true) ** 2, axis=1))
        dE = energy_function(y) - energy_function(x)

        red_jacobian = tf.reduce_sum(j_x, axis=1)

        dF = tf.reduce_mean(tf.abs(dE + red_jacobian))

        return factor_supervision * red_diff + dF

    def supervised_acceptance(
        self, energy_function="soft", factor_supervision=1000, factor_entropy=1
    ):
        if energy_function is None:
            energy_function = self.system.energy_tf
        if energy_function is "soft":
            energy_function = self.system.energy_hybr_tf
        return wrapped_partial(
            self._supervised_acceptance,
            energy_function=energy_function,
            factor_supervision=factor_supervision,
            factor_entropy=factor_entropy,
        )

    def __supervised_acceptance_init(
        self, y_true, y_pred, energy_function, factor_supervision, factor_entropy
    ):
        x, y, z, j_x, j_y = self.nn.split_output(y_pred)

        red_diff = tf.reduce_sum((y - y_true) ** 2, axis=1)
        dE = tf.abs(energy_function(y) - energy_function(x))

        red_jacobian = tf.abs(tf.reduce_sum(j_x, axis=1))

        return tf.reduce_mean(
            factor_supervision * red_diff + factor_entropy * red_jacobian + dE, axis=0
        )

    def _supervised_acceptance_init(
        self, y_true, y_pred, energy_function, factor_supervision, factor_entropy
    ):
        x, y, z, j_x, j_y = self.nn.split_output(y_pred)

        red_diff = tf.reduce_sum((y - y_true) ** 2, axis=1)
        dE = energy_function(y) - energy_function(x)

        red_jacobian = tf.reduce_sum(j_x, axis=1)

        dF = tf.reduce_mean(tf.abs(dE + red_jacobian))

        return tf.reduce_mean(factor_supervision * red_diff + dF, axis=0)

    def supervised_acceptance_init(
        self, energy_function="soft", factor_supervision=1000, factor_entropy=1
    ):
        if energy_function is None:
            energy_function = self.system.energy_tf
        if energy_function is "soft":
            energy_function = self.system.energy_hybr_tf
        return wrapped_partial(
            self._supervised_acceptance_init,
            energy_function=energy_function,
            factor_supervision=factor_supervision,
            factor_entropy=factor_entropy,
        )
