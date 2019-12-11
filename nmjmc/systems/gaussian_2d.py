import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

default_minima = np.array(
    [[2.0, 0.0], [-2.0, 0.0], [2.0, 0.6], [1.7, -0.6], [-2.0, 0.4], [-1.6, -1.1]]
)
default_sigmas = np.array(
    [[0.5, 1.2], [0.5, 1.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2]]
)
default_factors = np.array([10.0, 10.0, 1.4, 1.4, 1.4, 1.4, 0.2, 0.2]) * 0.5


class GaussianDoublewell:
    def __init__(
        self,
        minima=default_minima,
        sigmas=default_sigmas,
        dmid=0.0,
        a=16.0,
        b=1.0,
        scalefactor=default_factors,
    ):
        self.dim = 2
        self.minima_x = minima[:, 0]
        self.minima_y = minima[:, 1]
        self.inv_sigmas_x = 1.0 / (2 * sigmas[:, 0] ** 2)
        self.inv_sigmas_y = 1.0 / (2 * sigmas[:, 1] ** 2)
        self.prefactor = scalefactor[:-2] / (2 * np.pi * sigmas[:, 0] * sigmas[:, 1])
        self.dmid = dmid
        self.a = a
        self.b = b
        self.scalefactor_doublewell_x = scalefactor[-2]
        self.scalefactor_doublewell_y = scalefactor[-1]

    def energy(self, x):
        xcomp = x[:, 0].reshape(len(x), 1)
        ycomp = x[:, 1].reshape(len(x), 1)
        E = np.sum(
            -self.prefactor
            * np.exp(
                -((xcomp - self.minima_x) ** 2) * self.inv_sigmas_x
                - (ycomp - self.minima_y) ** 2 * self.inv_sigmas_y
            ),
            axis=1,
        )
        d = np.abs(xcomp)
        d2 = d * d
        d4 = d2 * d2
        E_doublewell = self.scalefactor_doublewell_x * (-self.a * d2 + self.b * d4)
        E_well_y = self.scalefactor_doublewell_y * ycomp ** 2
        return E + E_doublewell.flatten() + E_well_y.flatten()

    def energy_tf(self, x):
        xcomp = tf.reshape(x[:, 0], shape=(-1, 1))
        ycomp = tf.reshape(x[:, 1], shape=(-1, 1))
        E = tf.reduce_sum(
            -self.prefactor
            * tf.exp(
                -((xcomp - self.minima_x) ** 2) * self.inv_sigmas_x
                - (ycomp - self.minima_y) ** 2 * self.inv_sigmas_y
            ),
            axis=1,
        )
        # d = tf.abs(tf.reshape(xcomp, shape=(-1)))
        d2 = xcomp * xcomp
        d4 = d2 * d2
        E_doublewell_x = self.scalefactor_doublewell_x * (-self.a * d2 + self.b * d4)
        E_well_y = self.scalefactor_doublewell_y * ycomp ** 2
        return (
            E
            + tf.reshape(E_doublewell_x, shape=(-1,))
            + tf.reshape(E_well_y, shape=(-1,))
        )

    def potential_in_grid(self, xmin, xmax, ymin, ymax):
        x = np.arange(xmin, xmax, (xmax - xmin) / 100.0)
        y = np.arange(ymin, ymax, (ymax - ymin) / 100.0)
        xx, yy = np.meshgrid(x, y)
        original_shape = xx.shape
        xf = xx.flatten()
        yf = yy.flatten()
        xx_yy = np.array([xf, yf]).transpose()
        E = self.energy(xx_yy)
        E = E.reshape(original_shape)
        return xx, yy, E

    def plot_contour(
        self, bounds=None, numcontour=None, boltzmann=False, axis=None, **kwargs
    ):
        # set default values for bounds and number of countours
        if bounds == None:
            bounds = [-3, 3, -3, 3, 0, 3]
        xmin = bounds[0]
        xmax = bounds[1]
        ymin = bounds[2]
        ymax = bounds[3]
        zmin = bounds[4]
        zmax = bounds[5]
        if numcontour == None:
            numcontour = 25
        # calculate and plot potential or gradient
        xx, yy, zz = self.potential_in_grid(xmin, xmax, ymin, ymax)
        if boltzmann:
            zz = np.exp(-zz)
        if axis is None:
            plt.contour(xx, yy, zz, numcontour, **kwargs)
            plt.axes().set_aspect("equal")
        else:
            axis.contour(xx, yy, zz, numcontour, **kwargs)


default_triple_minima = np.array([[-2.2, -1.0], [0.0, 2], [2, -0.8]])
default_triple_sigmas = np.array([[0.5, 0.3], [0.5, 0.4], [0.4, 0.5]])
default_triple_scalefactors = np.array([5.0, 5.0, 5.0, 0.1])


class GaussianTripleWell(GaussianDoublewell):
    def __init__(
        self,
        minima=default_triple_minima,
        sigmas=default_triple_sigmas,
        scalefactors=default_triple_scalefactors,
    ):
        self.scalefactors = scalefactors[:-1]
        self.scalefactos_confinement = scalefactors[-1]
        self.prefactor = scalefactors[:-1] / (2 * np.pi * sigmas[:, 0] * sigmas[:, 1])
        self.inv_sigmas_x = 1.0 / (2 * sigmas[:, 0] ** 2)
        self.inv_sigmas_y = 1.0 / (2 * sigmas[:, 1] ** 2)
        self.minima_x = minima[:, 0]
        self.minima_y = minima[:, 1]
        self.dim = 2

    def confinement_potentials(self, x):
        r = np.linalg.norm(x, axis=1)
        return self.scalefactos_confinement * r ** 2

    def confinement_potentials_tf(self, x):
        r = x[:, 0] ** 2 + x[:, 1] ** 2
        return self.scalefactos_confinement * r

    def energy(self, x):
        xcomp = x[:, 0].reshape(len(x), 1)
        ycomp = x[:, 1].reshape(len(x), 1)
        E = np.sum(
            -self.prefactor
            * np.exp(
                -((xcomp - self.minima_x) ** 2) * self.inv_sigmas_x
                - (ycomp - self.minima_y) ** 2 * self.inv_sigmas_y
            ),
            axis=1,
        )
        return E + self.confinement_potentials(x)

    def energy_tf(self, x):
        xcomp = tf.reshape(x[:, 0], shape=(-1, 1))
        ycomp = tf.reshape(x[:, 1], shape=(-1, 1))
        E = tf.reduce_sum(
            -self.prefactor
            * tf.exp(
                -((xcomp - self.minima_x) ** 2) * self.inv_sigmas_x
                - (ycomp - self.minima_y) ** 2 * self.inv_sigmas_y
            ),
            axis=1,
        )
        return E + self.confinement_potentials_tf(x)
