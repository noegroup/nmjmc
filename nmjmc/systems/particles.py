import numpy as np
import tensorflow as tf


class Particles:
    def __init__(
        self,
        d=4.0,
        rm=1.1,
        nsolvent=36,
        a=3.0,
        b=1.1,
        c=1.0,
        eps=1.0,
        k=1.0,
        dmid=1.5,
        rc=0.8,
    ):
        self.d = d
        self.rm = rm
        self.nsolvent = nsolvent
        self.n = nsolvent + 2
        self.a = a
        self.b = b
        self.c = c
        self.eps = eps
        self.k = k
        self.dmid = dmid
        self.rc = rc
        # parameters for the inverse parabola
        self.a_harmonic = self.eps * (
            7.0 * (self.rm / self.rc) ** 12 - 8 * (self.rm / self.rc) ** 6
        )
        self.b_harmonic = (
            6.0
            * self.eps
            / self.rc ** 2
            * (self.rm / self.rc) ** 6
            * ((self.rm / self.rc) ** 6 - 1)
        )
        # parameters for the parabola
        self.c_harmonic = (
            -6 * (7 * eps * rc ** 6 * rm ** 6 - 13 * eps * rm ** 12) / rc ** 14
        )
        self.d_harmonic = (
            -2 * (4 * rc ** 7 - 7 * rc * rm ** 6) / (7 * rc ** 6 - 13 * rm ** 6)
        )
        self.e_harmonic = (
            -8 * eps * rc ** 12 * rm ** 6
            + 21 * eps * rc ** 6 * rm ** 12
            - 7 * eps * rm ** 18
        ) / (rc ** 12 * (7 * rc ** 6 - 13 * rm ** 6))
        self.mask_matrix = np.ones((self.n, self.n), dtype=np.float32)
        self.mask_matrix[0, 1] = 0.0
        self.mask_matrix[1, 0] = 0.0
        for i in range(self.n):
            self.mask_matrix[i, i] = 0.0
        self.dim = 2 * self.n

    def harmonic_energy(self, x, k=1.0, r=1.0):
        # all component-wise distances bet
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = np.shape(x)[0]
        n = np.shape(xcomp)[1]
        Xcomp = np.tile(np.expand_dims(xcomp, 2), (1, 1, n))
        Ycomp = np.tile(np.expand_dims(ycomp, 2), (1, 1, n))
        Dx = Xcomp - np.transpose(Xcomp, axes=(0, 2, 1))
        Dy = Ycomp - np.transpose(Ycomp, axes=(0, 2, 1))
        D2 = Dx ** 2 + Dy ** 2
        # mask_matrix = np.ones((n, n), dtype=np.float32)
        # for i in range(n):
        #     mask_matrix[i, i] = 0.0
        mmatrix = np.tile(np.expand_dims(self.mask_matrix, 0), (batchsize, 1, 1))
        D = (D2) ** 0.5
        compare_solv = r > D
        E = 0.5 * np.sum(k * (r - D) ** 2 * compare_solv * mmatrix, axis=(1, 2))
        return E

    def harmonic_energy_tf(self, x, k=1.0, r=1.0):
        # all component-wise distances bet
        xcomp = tf.to_float(x[:, 0::2])
        ycomp = tf.to_float(x[:, 1::2])
        batchsize = tf.shape(x)[0]
        n = tf.shape(xcomp)[1]
        Xcomp = tf.tile(tf.expand_dims(xcomp, 2), [1, 1, n])
        Ycomp = tf.tile(tf.expand_dims(ycomp, 2), [1, 1, n])
        Dx = Xcomp - tf.transpose(Xcomp, perm=[0, 2, 1])
        Dy = Ycomp - tf.transpose(Ycomp, perm=[0, 2, 1])
        D2 = Dx ** 2 + Dy ** 2 + tf.eye(n)
        D = tf.sqrt(D2) - tf.eye(n)
        compare_solv = tf.to_float(r > D)
        self_energy = 0.5 * k * r * r * tf.to_float(n)
        E = (
            0.5 * tf.reduce_sum(k * (r - D) ** 2 * compare_solv, axis=(1, 2))
            - self_energy
        )
        return E

    def _LJ_energy(self, D2):
        return 0.5 * self.eps * np.sum(D2 ** 6 - 2 * D2 ** 3, axis=(1, 2))

    def _LJ_energy_tf(self, D2):
        return 0.5 * self.eps * tf.reduce_sum(D2 ** 6 - 2 * D2 ** 3, axis=(1, 2))

    def _harmonic_energy(self, D2):
        return 0.5 * np.sum(self.a_harmonic - self.b_harmonic * D2, axis=(1, 2))

    def _harmonic_energy_tf(self, D2):
        return 0.5 * tf.reduce_sum(self.a_harmonic - self.b_harmonic * D2, axis=(1, 2))

    def LJ_energy(self, x):
        # all component-wise distances bet
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = np.shape(x)[0]
        n = np.shape(xcomp)[1]
        Xcomp = np.tile(np.expand_dims(xcomp, 2), (1, 1, n))
        Ycomp = np.tile(np.expand_dims(ycomp, 2), (1, 1, n))
        Dx = Xcomp - np.transpose(Xcomp, axes=(0, 2, 1))
        Dy = Ycomp - np.transpose(Ycomp, axes=(0, 2, 1))
        D2 = Dx ** 2 + Dy ** 2
        # mask_matrix = np.ones((n, n), dtype=np.float32)
        # mask_matrix[0, 1] = 0.0
        # mask_matrix[1, 0] = 0.0
        # for i in range(n):
        #     mask_matrix[i, i] = 0.0
        mmatrix = np.tile(np.expand_dims(self.mask_matrix, 0), (batchsize, 1, 1))
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = (self.rm ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix
        # energy
        # do 1/2 because we have double-counted each interaction
        # E = 0.5 * self.eps * np.sum(D2rel ** 6 - 2 * D2rel ** 3, axis=(1, 2))
        return self._LJ_energy(D2rel)

    def WCA_energy(self, x):
        # all component-wise distances bet
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = np.shape(x)[0]
        n = np.shape(xcomp)[1]
        Xcomp = np.tile(np.expand_dims(xcomp, 2), (1, 1, n))
        Ycomp = np.tile(np.expand_dims(ycomp, 2), (1, 1, n))
        Dx = Xcomp - np.transpose(Xcomp, axes=(0, 2, 1))
        Dy = Ycomp - np.transpose(Ycomp, axes=(0, 2, 1))
        D2 = Dx ** 2 + Dy ** 2
        # mask_matrix = np.ones((n, n), dtype=np.float32)
        # mask_matrix[0, 1] = 0.0
        # mask_matrix[1, 0] = 0.0
        # for i in range(n):
        #     mask_matrix[i, i] = 0.0
        mmatrix = np.tile(np.expand_dims(self.mask_matrix, 0), (batchsize, 1, 1))
        distance_mask = D2 < self.rm ** 2
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = (self.rm ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix * distance_mask
        # energy
        # do 1/2 because we have double-counted each interaction
        E = (
            0.5
            * self.eps
            * np.sum(
                D2rel ** 6 - 2 * D2rel ** 3 + (mmatrix) * (distance_mask), axis=(1, 2)
            )
        )
        return E

    def WCA_energy_tf(self, x):
        # all component-wise distances bet
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = tf.shape(x)[0]
        n = tf.shape(xcomp)[1]
        Xcomp = tf.tile(tf.expand_dims(xcomp, 2), [1, 1, n])
        Ycomp = tf.tile(tf.expand_dims(ycomp, 2), [1, 1, n])
        Dx = Xcomp - tf.transpose(Xcomp, perm=[0, 2, 1])
        Dy = Ycomp - tf.transpose(Ycomp, perm=[0, 2, 1])
        D2 = Dx ** 2 + Dy ** 2
        mmatrix = tf.tile(tf.expand_dims(self.mask_matrix, 0), [batchsize, 1, 1])
        dtype = D2.dtype
        distance_mask = tf.cast(D2 < self.rm ** 2, dtype)
        mmatrix = tf.cast(mmatrix, dtype)
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = (self.rm ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix * distance_mask
        # energy
        # do 1/2 because we have double-counted each interaction
        E = (
            0.5
            * self.eps
            * tf.reduce_sum(
                D2rel ** 6 - 2 * D2rel ** 3 + mmatrix * distance_mask, axis=(1, 2)
            )
        )
        return E

    def LJ_energy_tf(self, x):
        # all component-wise distances bet
        # this is optimized to only work for n=38 particles
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = tf.shape(x)[0]
        n = tf.shape(xcomp)[1]
        Xcomp = tf.tile(tf.expand_dims(xcomp, 2), [1, 1, n])
        Ycomp = tf.tile(tf.expand_dims(ycomp, 2), [1, 1, n])
        Dx = Xcomp - tf.transpose(Xcomp, perm=[0, 2, 1])
        Dy = Ycomp - tf.transpose(Ycomp, perm=[0, 2, 1])
        D2 = Dx ** 2 + Dy ** 2
        mmatrix = tf.tile(tf.expand_dims(self.mask_matrix, 0), [batchsize, 1, 1])
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = (self.rm ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix
        # energy
        # do 1/2 because we have double-counted each interaction
        E = 0.5 * self.eps * tf.reduce_sum(D2rel ** 6 - 2 * D2rel ** 3, axis=(1, 2))
        return E

    def _LJ_energy_soft(self, x):
        # all component-wise distances bet
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = np.shape(x)[0]
        n = np.shape(xcomp)[1]
        Xcomp = np.tile(np.expand_dims(xcomp, 2), (1, 1, n))
        Ycomp = np.tile(np.expand_dims(ycomp, 2), (1, 1, n))
        Dx = Xcomp - np.transpose(Xcomp, axes=(0, 2, 1))
        Dy = Ycomp - np.transpose(Ycomp, axes=(0, 2, 1))
        D2 = Dx ** 2 + Dy ** 2
        distance_mask = D2 > self.rc ** 2
        # Set distance to higher value to avoid numerical problems. This is set to zero in the energy later.
        # mask_matrix = np.ones((n, n), dtype=np.float32)
        # mask_matrix[0, 1] = 0.0
        # mask_matrix[1, 0] = 0.0
        # for i in range(n):
        #     mask_matrix[i, i] = 0.0
        E_h = (
            (self.a_harmonic - self.b_harmonic * D2)
            * (1.0 - distance_mask)
            * self.mask_matrix
        )
        mmatrix = np.tile(np.expand_dims(self.mask_matrix, 0), (batchsize, 1, 1))
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = (self.rm ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix * distance_mask
        return self._LJ_energy(D2rel) + 0.5 * np.sum(E_h, axis=(1, 2))

    def LJ_energy_soft(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = np.shape(x)[0]
        n = np.shape(xcomp)[1]
        Xcomp = np.tile(np.expand_dims(xcomp, 2), (1, 1, n))
        Ycomp = np.tile(np.expand_dims(ycomp, 2), (1, 1, n))
        Dx = Xcomp - np.transpose(Xcomp, axes=(0, 2, 1))
        Dy = Ycomp - np.transpose(Ycomp, axes=(0, 2, 1))
        D2 = Dx ** 2 + Dy ** 2
        distance_mask = D2 > self.rc ** 2
        # Set distance to higher value to avoid numerical problems. This is set to zero in the energy later.
        # mask_matrix = np.ones((n, n), dtype=np.float32)
        # mask_matrix[0, 1] = 0.0
        # mask_matrix[1, 0] = 0.0
        # for i in range(n):
        #     mask_matrix[i, i] = 0.0
        E_h = (
            (self.c_harmonic * (np.sqrt(D2) + self.d_harmonic) ** 2 + self.e_harmonic)
            * (1 - distance_mask)
            * self.mask_matrix
        )
        mmatrix = np.tile(np.expand_dims(self.mask_matrix, 0), (batchsize, 1, 1))
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = (self.rm ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix * distance_mask
        return self._LJ_energy(D2rel) + 0.5 * np.sum(E_h, axis=(1, 2))

    def _LJ_energy_soft_tf(self, x):
        # all component-wise distances bet
        # this is optimized to only work for n=38 particles
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = tf.shape(x)[0]
        n = tf.shape(xcomp)[1]
        Xcomp = tf.tile(tf.expand_dims(xcomp, 2), [1, 1, n])
        Ycomp = tf.tile(tf.expand_dims(ycomp, 2), [1, 1, n])
        Dx = Xcomp - tf.transpose(Xcomp, perm=[0, 2, 1])
        Dy = Ycomp - tf.transpose(Ycomp, perm=[0, 2, 1])
        D2 = Dx ** 2 + Dy ** 2
        dtype = D2.dtype
        distance_mask = tf.cast(D2 > self.rc ** 2, dtype)
        mask_matrix = tf.cast(self.mask_matrix, dtype)
        mmatrix = tf.tile(tf.expand_dims(mask_matrix, 0), [batchsize, 1, 1])
        E_h = (self.a_harmonic - self.b_harmonic * D2) * (1.0 - distance_mask) * mmatrix
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = (self.rm ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix * distance_mask
        # energy
        # do 1/2 because we have double-counted each interaction
        return self._LJ_energy_tf(D2rel) + 0.5 * tf.reduce_sum(E_h, axis=(1, 2))

    def LJ_energy_soft_tf(self, x):
        # all component-wise distances bet
        # this is optimized to only work for n=38 particles
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = tf.shape(x)[0]
        n = np.shape(xcomp)[1]
        Xcomp = tf.tile(tf.expand_dims(xcomp, 2), [1, 1, n])
        Ycomp = tf.tile(tf.expand_dims(ycomp, 2), [1, 1, n])
        Dx = Xcomp - tf.transpose(Xcomp, perm=[0, 2, 1])
        Dy = Ycomp - tf.transpose(Ycomp, perm=[0, 2, 1])
        D2 = Dx ** 2 + Dy ** 2
        dtype = D2.dtype
        distance_mask = tf.cast(D2 > self.rc ** 2, dtype)
        mask_matrix = tf.cast(self.mask_matrix, dtype)
        mmatrix = tf.tile(tf.expand_dims(mask_matrix, 0), [batchsize, 1, 1])
        E_h = (
            (
                self.c_harmonic
                * (tf.sqrt(D2 + (1.0 - mask_matrix)) + self.d_harmonic) ** 2
                + self.e_harmonic
            )
            * (1 - distance_mask)
            * mask_matrix
        )
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = (self.rm ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix * distance_mask
        # energy
        # do 1/2 because we have double-counted each interaction
        return self._LJ_energy_tf(D2rel) + 0.5 * tf.reduce_sum(E_h, axis=(1, 2))

    def particle_softcore_harm_tf(self, x, exp=2):
        # all component-wise distances bet
        # this is optimized to only w`ork for n=38 particles
        xsolv = x[:, 4:]
        xcomp = xsolv[:, 0::2]
        ycomp = xsolv[:, 1::2]
        batchsize = tf.shape(x)[0]
        n = tf.shape(xcomp)[1]
        Xcomp = tf.tile(tf.expand_dims(xcomp, 2), [1, 1, n])
        Ycomp = tf.tile(tf.expand_dims(ycomp, 2), [1, 1, n])

        int_length_dim = 1
        int_length_solv = 1

        Dx = Xcomp - tf.transpose(Xcomp, perm=[0, 2, 1])
        Dy = Ycomp - tf.transpose(Ycomp, perm=[0, 2, 1])
        D2_solv = Dx ** 2 + Dy ** 2
        compare_solv = tf.to_float(D2_solv < int_length_solv)
        mask_matrix_solvent = self.mask_matrix[2:, 2:]
        mmatrix = tf.tile(tf.expand_dims(mask_matrix_solvent, 0), [batchsize, 1, 1])
        D2_solv = D2_solv * mmatrix + tf.eye(1)
        D2_solv = tf.sqrt(D2_solv) - tf.eye(1)
        E_solv = tf.reduce_sum(
            ((int_length_solv - D2_solv) ** exp * compare_solv), axis=(1, 2)
        )

        Dx_dim1 = xcomp[:, :] - tf.tile(
            tf.expand_dims(x[:, 0], axis=1), [1, self.nsolvent]
        )
        Dy_dim1 = ycomp[:, :] - tf.tile(
            tf.expand_dims(x[:, 1], axis=1), [1, self.nsolvent]
        )
        Dx_dim2 = xcomp[:, :] - tf.tile(
            tf.expand_dims(x[:, 2], axis=1), [1, self.nsolvent]
        )
        Dy_dim2 = ycomp[:, :] - tf.tile(
            tf.expand_dims(x[:, 3], axis=1), [1, self.nsolvent]
        )
        D2_dim1 = tf.sqrt(Dx_dim1 ** 2 + Dy_dim1 ** 2)
        D2_dim2 = tf.sqrt(Dx_dim2 ** 2 + Dy_dim2 ** 2)
        compare_dim1 = tf.to_float(D2_dim1 < int_length_dim)
        compare_dim2 = tf.to_float(D2_dim2 < int_length_dim)
        E_dim1 = tf.reduce_sum(
            (int_length_dim - D2_dim1) ** exp * compare_dim1, axis=(1)
        )
        E_dim2 = tf.reduce_sum(
            (int_length_dim - D2_dim2) ** exp * compare_dim2, axis=(1)
        )
        E_dim = E_dim1 + E_dim2

        E = E_solv + E_dim
        return E

    def box_energy_morse(self, x, De=1.0, a_tem=6.0):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        a = a_tem / self.rm
        E = 0.0
        dxl2 = (xcomp - self.d) ** 2
        Rxl2 = np.sqrt(dxl2 + 0.001)
        E += De * (1.0 - np.exp(-a * (Rxl2 - self.rm))) ** 2 - De
        dxr2 = (xcomp + self.d) ** 2
        Rxr2 = np.sqrt(dxr2 + 0.001)
        E += De * (1.0 - np.exp(-a * (Rxr2 - self.rm))) ** 2 - De
        dyl2 = (ycomp - self.d) ** 2
        Ryl2 = np.sqrt(dyl2 + 0.001)
        E += De * (1.0 - np.exp(-a * (Ryl2 - self.rm))) ** 2 - De
        dyr2 = (ycomp + self.d) ** 2
        Ryr2 = np.sqrt(dyr2 + 0.001)
        E += De * (1.0 - np.exp(-a * (Ryr2 - self.rm))) ** 2 - De
        return np.sum(E, axis=1)

    def box_energy_morse_tf(self, x, De=1.0, a_tem=6.0):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        a = a_tem / self.rm
        E = 0.0
        dxl2 = (xcomp - self.d) ** 2
        Rxl2 = tf.sqrt(dxl2 + 0.001)
        E += De * (1.0 - tf.exp(-a * (Rxl2 - self.rm))) ** 2 - De
        dxr2 = (xcomp + self.d) ** 2
        Rxr2 = tf.sqrt(dxr2 + 0.001)
        E += De * (1.0 - tf.exp(-a * (Rxr2 - self.rm))) ** 2 - De
        dyl2 = (ycomp - self.d) ** 2
        Ryl2 = tf.sqrt(dyl2 + 0.001)
        E += De * (1.0 - tf.exp(-a * (Ryl2 - self.rm))) ** 2 - De
        dyr2 = (ycomp + self.d) ** 2
        Ryr2 = tf.sqrt(dyr2 + 0.001)
        E += De * (1.0 - tf.exp(-a * (Ryr2 - self.rm))) ** 2 - De

        return tf.reduce_sum(E, axis=1)

    def LJ_energy_hybr_tf(self, x, De=1.0, a_tem=6.0):
        batchsize = tf.shape(x)[0]

        a = a_tem / self.rm

        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]

        Xcomp = tf.tile(tf.expand_dims(xcomp, 2), [1, 1, self.n])
        Ycomp = tf.tile(tf.expand_dims(ycomp, 2), [1, 1, self.n])
        Dx = Xcomp - tf.transpose(Xcomp, perm=[0, 2, 1])
        Dy = Ycomp - tf.transpose(Ycomp, perm=[0, 2, 1])
        D2 = Dx ** 2 + Dy ** 2

        mmatrix = tf.tile(tf.expand_dims(self.mask_matrix, 0), [batchsize, 1, 1])
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later

        D2rel = (self.rm ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix
        # energy

        E_LJ = 0.5 * tf.reduce_sum(
            self.eps * (D2rel ** 6 - 2 * D2rel ** 3), axis=(1, 2)
        )

        R = tf.sqrt(
            D2 + 0.001
        )  # add a little bit to prevent the gradient from blowing up
        E_morse = tf.reduce_sum(
            (De * (1.0 - tf.exp(-a * (R - self.rm))) ** 2 - De), axis=(1, 2)
        )

        check = tf.is_finite(1000000 * E_LJ)
        E = tf.where(check, E_LJ, E_morse)

        return E

    def box_energy_hybr_tf(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]

        E_LJ = 0.0
        dxl2 = self.rm ** 2 / (xcomp - self.d) ** 2
        E_LJ += self.eps * tf.reduce_sum(dxl2 ** 6 - 2 * dxl2 ** 3, axis=1)
        dxr2 = self.rm ** 2 / (xcomp + self.d) ** 2
        E_LJ += self.eps * tf.reduce_sum(dxr2 ** 6 - 2 * dxr2 ** 3, axis=1)
        dyl2 = self.rm ** 2 / (ycomp - self.d) ** 2
        E_LJ += self.eps * tf.reduce_sum(dyl2 ** 6 - 2 * dyl2 ** 3, axis=1)
        dyr2 = self.rm ** 2 / (ycomp + self.d) ** 2
        E_LJ += self.eps * tf.reduce_sum(dyr2 ** 6 - 2 * dyr2 ** 3, axis=1)

        E_exp = 0.0
        dxl2 = (xcomp - self.d) ** 2
        Rxl2 = tf.sqrt(dxl2 + 0.001)
        E_exp += tf.reduce_sum(tf.exp(-Rxl2), axis=1)
        dxr2 = (xcomp + self.d) ** 2
        Rxr2 = tf.sqrt(dxr2 + 0.001)
        E_exp += tf.reduce_sum(tf.exp(-Rxr2), axis=1)
        dyl2 = (ycomp - self.d) ** 2
        Ryl2 = tf.sqrt(dyl2 + 0.001)
        E_exp += tf.reduce_sum(tf.exp(-Ryl2), axis=1)
        dyr2 = (ycomp + self.d) ** 2
        Ryr2 = tf.sqrt(dyr2 + 0.001)
        E_exp += tf.reduce_sum(tf.exp(-Ryr2), axis=1)

        check = tf.is_finite(1000000 * E_LJ)
        E = tf.where(check, E_LJ, E_exp)
        return E

    def morse_energy_tf(self, x, De=1.0, a=6.0):
        # all component-wise distances bet
        # this is optimized to only work for n=38 particles
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = tf.shape(x)[0]
        n = tf.shape(xcomp)[1]
        Xcomp = tf.tile(tf.expand_dims(xcomp, 2), [1, 1, n])
        Ycomp = tf.tile(tf.expand_dims(ycomp, 2), [1, 1, n])
        Dx = Xcomp - tf.transpose(Xcomp, perm=[0, 2, 1])
        Dy = Ycomp - tf.transpose(Ycomp, perm=[0, 2, 1])
        D2 = Dx ** 2 + Dy ** 2
        mmatrix = tf.tile(tf.expand_dims(self.mask_matrix, 0), [batchsize, 1, 1])
        R = tf.sqrt(D2)
        E = De * (1.0 - tf.exp(-a * (R - self.rm))) ** 2 - De
        return 0.5 * tf.reduce_sum(E * mmatrix, axis=(1, 2))

    def dimer_energy(self, x, dmid=1.5):
        k_restraint = 20.0
        # center restraint energy
        energy_dx = k_restraint * (x[:, 0] + x[:, 2]) ** 2
        # y restraint energy
        energy_dy = k_restraint * (x[:, 1]) ** 2 + k_restraint * (x[:, 3]) ** 2
        # first two particles
        d = np.sqrt((x[:, 0] - x[:, 2]) ** 2 + (x[:, 1] - x[:, 3]) ** 2)
        d0 = 2 * (d - self.dmid)
        d2 = d0 * d0
        d4 = d2 * d2
        energy_interaction = self.c * (-self.a * d2 + self.b * d4)

        return energy_dx + energy_dy + energy_interaction

    def dimer_energy_tf(self, x, dmid=1.5):
        k_restraint = 20.0
        # center restraint energy
        energy_dx = k_restraint * (x[:, 0] + x[:, 2]) ** 2
        # y restraint energy
        energy_dy = k_restraint * (x[:, 1]) ** 2 + k_restraint * (x[:, 3]) ** 2
        # first two particles
        d = tf.sqrt((x[:, 0] - x[:, 2]) ** 2 + (x[:, 1] - x[:, 3]) ** 2)
        d0 = 2 * (d - self.dmid)
        d2 = d0 * d0
        d4 = d2 * d2
        energy_interaction = self.c * (-self.a * d2 + self.b * d4)

        return energy_dx + energy_dy + energy_interaction

    def box_energy(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        E = 0.0
        dxl2 = self.rm ** 2 / (xcomp - self.d) ** 2
        E += self.eps * np.sum(dxl2 ** 6 - 2 * dxl2 ** 3, axis=1)
        dxr2 = self.rm ** 2 / (xcomp + self.d) ** 2
        E += self.eps * np.sum(dxr2 ** 6 - 2 * dxr2 ** 3, axis=1)
        dyl2 = self.rm ** 2 / (ycomp - self.d) ** 2
        E += self.eps * np.sum(dyl2 ** 6 - 2 * dyl2 ** 3, axis=1)
        dyr2 = self.rm ** 2 / (ycomp + self.d) ** 2
        E += self.eps * np.sum(dyr2 ** 6 - 2 * dyr2 ** 3, axis=1)
        return E

    def box_energy_wca(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        E = 0.0
        mask_xl = (xcomp - self.d) ** 2 < self.rm ** 2
        dxl2 = self.rm ** 2 / (xcomp - self.d) ** 2 * mask_xl
        E += self.eps * np.sum(dxl2 ** 6 - 2 * dxl2 ** 3 + mask_xl, axis=1)
        mask_xr = (xcomp + self.d) ** 2 < self.rm ** 2
        dxr2 = self.rm ** 2 / (xcomp + self.d) ** 2 * mask_xr
        E += self.eps * np.sum(dxr2 ** 6 - 2 * dxr2 ** 3 + mask_xr, axis=1)
        mask_yl = (ycomp - self.d) ** 2 < self.rm ** 2
        dyl2 = self.rm ** 2 / (ycomp - self.d) ** 2 * mask_yl
        E += self.eps * np.sum(dyl2 ** 6 - 2 * dyl2 ** 3 + mask_yl, axis=1)
        mask_yr = (ycomp + self.d) ** 2 < self.rm ** 2
        dyr2 = self.rm ** 2 / (ycomp + self.d) ** 2 * mask_yr
        E += self.eps * np.sum(dyr2 ** 6 - 2 * dyr2 ** 3 + mask_yr, axis=1)
        return E

    def box_energy_wca_tf(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        E = 0.0
        dtype = x.dtype
        mask_xl = tf.cast((xcomp - self.d) ** 2 < self.rm ** 2, dtype)
        dxl2 = self.rm ** 2 / (xcomp - self.d) ** 2 * mask_xl
        E += self.eps * tf.reduce_sum(
            dxl2 ** 6 - 2 * dxl2 ** 3 + self.eps * mask_xl, axis=1
        )
        mask_xr = tf.cast((xcomp + self.d) ** 2 < self.rm ** 2, dtype)
        dxr2 = self.rm ** 2 / (xcomp + self.d) ** 2 * mask_xr
        E += self.eps * tf.reduce_sum(
            dxr2 ** 6 - 2 * dxr2 ** 3 + self.eps * mask_xr, axis=1
        )
        mask_yl = tf.cast((ycomp - self.d) ** 2 < self.rm ** 2, dtype)
        dyl2 = self.rm ** 2 / (ycomp - self.d) ** 2 * mask_yl
        E += self.eps * tf.reduce_sum(
            dyl2 ** 6 - 2 * dyl2 ** 3 + self.eps * mask_yl, axis=1
        )
        mask_yr = tf.cast((ycomp + self.d) ** 2 < self.rm ** 2, dtype)
        dyr2 = self.rm ** 2 / (ycomp + self.d) ** 2 * mask_yr
        E += self.eps * tf.reduce_sum(
            dyr2 ** 6 - 2 * dyr2 ** 3 + self.eps * mask_yr, axis=1
        )
        return E

    def box_energy_soft(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        E = 0.0
        # right wall of box
        xl = self.d - xcomp
        mask_dxl = xl < self.rc
        dxl2 = self.rm ** 2 / (xl + mask_dxl) ** 2
        dxl2 *= 1.0 - mask_dxl  # only keep part that is larger than the cutoff
        E += self.eps * np.sum(dxl2 ** 6 - 2 * dxl2 ** 3, axis=1)
        E += np.sum((self.a_harmonic - self.b_harmonic * xl ** 2) * mask_dxl, axis=1)

        # left wall of box
        xr = xcomp + self.d
        mask_dxr = xr < self.rc
        dxr2 = self.rm ** 2 / (xr + mask_dxr) ** 2
        dxr2 *= 1.0 - mask_dxr  # only keep part that is larger than the cutoff
        E += self.eps * np.sum(dxr2 ** 6 - 2 * dxr2 ** 3, axis=1)
        E += np.sum((self.a_harmonic - self.b_harmonic * (xr) ** 2) * mask_dxr, axis=1)

        # top wall of box
        yl = self.d - ycomp
        mask_dyl = yl < self.rc
        dyl2 = self.rm ** 2 / (yl + mask_dyl) ** 2
        dyl2 *= 1.0 - mask_dyl  # only keep part that is larger than the cutoff
        E += self.eps * np.sum(dyl2 ** 6 - 2 * dyl2 ** 3, axis=1)
        E += np.sum((self.a_harmonic - self.b_harmonic * (yl) ** 2) * mask_dyl, axis=1)

        # bottom wall of box
        yr = ycomp + self.d
        mask_dyr = yr < self.rc
        dyr2 = self.rm ** 2 / (yr + mask_dyr) ** 2
        dyr2 *= 1.0 - mask_dyr  # only keep part that is larger than the cutoff
        E += self.eps * np.sum(dyr2 ** 6 - 2 * dyr2 ** 3, axis=1)
        E += np.sum((self.a_harmonic - self.b_harmonic * (yr) ** 2) * mask_dyr, axis=1)
        return E

    def box_energy_tf(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        E = 0.0
        dxl2 = self.rm ** 2 / (xcomp - self.d) ** 2
        E += self.eps * tf.reduce_sum(dxl2 ** 6 - 2 * dxl2 ** 3, axis=1)
        dxr2 = self.rm ** 2 / (xcomp + self.d) ** 2
        E += self.eps * tf.reduce_sum(dxr2 ** 6 - 2 * dxr2 ** 3, axis=1)
        dyl2 = self.rm ** 2 / (ycomp - self.d) ** 2
        E += self.eps * tf.reduce_sum(dyl2 ** 6 - 2 * dyl2 ** 3, axis=1)
        dyr2 = self.rm ** 2 / (ycomp + self.d) ** 2
        E += self.eps * tf.reduce_sum(dyr2 ** 6 - 2 * dyr2 ** 3, axis=1)
        return E

    def box_energy_soft_tf(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        E = 0.0
        # right wall of box
        xl = self.d - xcomp
        mask_dxl = xl < self.rc
        dxl2 = self.rm ** 2 / (xl + mask_dxl) ** 2
        dxl2 *= 1.0 - mask_dxl  # only keep part that is larger than the cutoff
        E += self.eps * np.sum(dxl2 ** 6 - 2 * dxl2 ** 3, axis=1)
        E += np.sum((self.a_harmonic - self.b_harmonic * xl ** 2) * mask_dxl, axis=1)

        # left wall of box
        xr = xcomp + self.d
        mask_dxr = xr < self.rc
        dxr2 = self.rm ** 2 / (xr + mask_dxr) ** 2
        dxr2 *= 1.0 - mask_dxr  # only keep part that is larger than the cutoff
        E += self.eps * np.sum(dxr2 ** 6 - 2 * dxr2 ** 3, axis=1)
        E += np.sum((self.a_harmonic - self.b_harmonic * (xr) ** 2) * mask_dxr, axis=1)

        # top wall of box
        yl = self.d - ycomp
        mask_dyl = yl < self.rc
        dyl2 = self.rm ** 2 / (yl + mask_dyl) ** 2
        dyl2 *= 1.0 - mask_dyl  # only keep part that is larger than the cutoff
        E += self.eps * np.sum(dyl2 ** 6 - 2 * dyl2 ** 3, axis=1)
        E += np.sum((self.a_harmonic - self.b_harmonic * (yl) ** 2) * mask_dyl, axis=1)

        # bottom wall of box
        yr = ycomp + self.d
        mask_dyr = yr < self.rc
        dyr2 = self.rm ** 2 / (yr + mask_dyr) ** 2
        dyr2 *= 1.0 - mask_dyr  # only keep part that is larger than the cutoff
        E += self.eps * np.sum(dyr2 ** 6 - 2 * dyr2 ** 3, axis=1)
        E += np.sum((self.a_harmonic - self.b_harmonic * (yr) ** 2) * mask_dyr, axis=1)
        return E

    def box_energy_mod(self, x, kbox=10):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        E = 0.0
        dxl2 = xcomp - self.d
        E += np.sum(np.exp(kbox * dxl2), axis=1)
        dxr2 = -xcomp - self.d
        E += np.sum(np.exp(kbox * dxr2), axis=1)
        dyl2 = -ycomp - self.d
        E += np.sum(np.exp(kbox * dyl2), axis=1)
        dyr2 = ycomp - self.d
        E += np.sum(np.exp(kbox * dyr2), axis=1)
        return E

    def box_energy_mod_tf(self, x, kbox=10):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        E = 0.0
        dxl2 = xcomp - self.d
        E += tf.reduce_sum(tf.exp(kbox * dxl2), axis=1)
        dxr2 = -xcomp - self.d
        E += tf.reduce_sum(tf.exp(kbox * dxr2), axis=1)
        dyl2 = -ycomp - self.d
        E += tf.reduce_sum(tf.exp(kbox * dyl2), axis=1)
        dyr2 = ycomp - self.d
        E += tf.reduce_sum(tf.exp(kbox * dyr2), axis=1)
        return E

    def LJ(self, x, eps=1.0, rm=1.1):
        return eps * ((rm / x) ** 12 - 2 * (rm / x) ** 6)

    def LJ_der(self, x, eps=1.0, rm=1.1):
        return eps * (-12 * (rm / x) ** 13 + 12 * (rm / x) ** 7)

    def energy_hybr_tf(self, x):
        return (
            self.LJ_energy_hybr_tf(x)
            + self.dimer_energy_tf(x)
            + self.box_energy_hybr_tf(x)
        )

    def energy_hybr_inter_tf(self, x):
        return self.LJ_energy_hybr_tf(x) + self.box_energy_hybr_tf(x)

    def energy(self, x):
        return self.LJ_energy(x) + self.dimer_energy(x) + self.box_energy(x)

    def energy_wca(self, x):
        return self.WCA_energy(x) + self.dimer_energy(x) + self.box_energy_wca(x)

    def energy_wca_tf(self, x):
        return (
            self.WCA_energy_tf(x) + self.dimer_energy_tf(x) + self.box_energy_wca_tf(x)
        )

    def energy_tf(self, x):
        return self.LJ_energy_tf(x) + self.dimer_energy_tf(x) + self.box_energy_tf(x)

    def energy_mod(self, x):
        return self.LJ_energy(x) + self.dimer_energy(x) + self.box_energy_mod(x)

    def energy_mod_tf(self, x):
        return (
            self.LJ_energy_tf(x) + self.dimer_energy_tf(x) + self.box_energy_mod_tf(x)
        )

    def energy_soft(self, x):
        return self.LJ_energy_soft(x) + self.dimer_energy(x) + self.box_energy_soft(x)

    def energy_soft_tf(self, x):
        return (
            self.LJ_energy_soft_tf(x)
            + self.dimer_energy_tf(x)
            + self.box_energy_morse_tf(x)
        )


class FullLJParticles(Particles):
    def __init__(
        self,
        d=4.0,
        rm=1.1,
        nsolvent=36,
        a=2.7,
        b=1.1,
        c=1.0,
        eps=1.0,
        k=1.0,
        dmid=1.65,
        rc=0.8,
    ):
        super().__init__(d, rm, nsolvent, a, b, c, eps, k, dmid, rc)
        self.mask_matrix = np.ones((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            self.mask_matrix[i, i] = 0.0
