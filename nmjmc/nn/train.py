import numpy as np
import tensorflow.keras as keras


class TrainNetworkBiased:
    def __init__(
        self,
        nn,
        system,
        bias,
        loss,
        confs_open,
        confs_closed,
        reference_open,
        reference_closed,
        validation_data,
    ):
        self.nn = nn
        self.system = system
        self.bias = bias
        self.loss = loss
        self.confs_open = confs_open
        self.confs_closed = confs_closed
        self.reference_open = reference_open
        self.reference_closed = reference_closed
        self.validation_data = validation_data

    def train_init(self, n_epochs, n_samples, lr=0.001):
        training_input = np.array(
            [self.reference_open] * n_samples + [self.reference_closed] * n_samples
        )
        training_labels = np.array(
            [self.reference_closed] * n_samples + [self.reference_open] * n_samples
        )
        loss_function = self.loss.initialization_loss(
            self.system.energy_hybr_tf, 1000.0, 1.0
        )
        self.nn.train_pair(
            training_input,
            training_labels,
            loss_function,
            nepochs=n_epochs,
            batchsize=512,
            learning_rate=lr,
        )

    def train(self, n_epochs, clipnorm, lr=0.0001, batchsize=512, temperature=1.0):
        training_input = np.concatenate([self.confs_open, self.confs_closed])
        dummy_labels = np.zeros_like(training_input)
        dummy_labels_validation = np.zeros_like(self.validation_data)
        validation = [self.validation_data, dummy_labels_validation]
        loss_function = self.loss.unsupervised_acceptance(
            self.system.energy_hybr_tf,
            self.bias.energy_tf,
            factor_bias=1.0,
            factor_temperature=temperature,
        )
        self.nn.train_pair(
            training_input,
            dummy_labels,
            loss_function,
            validation_data=validation,
            nepochs=n_epochs,
            batchsize=batchsize,
            learning_rate=lr,
            clipnorm=clipnorm,
        )
