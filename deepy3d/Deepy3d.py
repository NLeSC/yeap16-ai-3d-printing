#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import keras as ks
from glob import glob
from sklearn.metrics import classification_report
from natsort import natsorted

from ConfigReader import ConfigReader
from PatientIO import PatientIO
from CNN_Model import CNN_Model
from Scans import Scans

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Deepy3d(object):
    """
    CT-image bone segmentation class.

    Calls a configuration file reader, reads in patient CT-scan data and trains
    a convolutional neural network. After training, it can perform bone
    segmentation on new patients and visualise its predictions, its learned
    kernels and the activations of the new scan in each layer.
    """

    def __init__(self, config_file):
        """Initialize Deepy3d class with a configuration file."""
        self.config = ConfigReader(config_file)

    def process_training_data(self):
        """Read and save the CT slices in another format."""
        print('* Reading CT scan files.')

        # Extract file directories
        trn_files = natsorted(self.config.get_CT_scans())
        lbl_files = natsorted(self.config.get_CT_labels())
        thr_files = natsorted(self.config.get_CT_thresholded())

        # Iterate over patients
        for i in range(self.config.get_num_patients()):

            # Call a PatientIO instance for the i-th file
            patient = PatientIO(trn_files[i])

            # Save training data (.png) and dir of training data
            patient.save_scan(i, self.config.get_trn_CT_slice_PNG_dir(),
                              self.config.get_trn_CT_slice_NPY_dir())

            # Save labels (.png) to label directory
            patient.read_save_labels(i, lbl_files[i],
                                     self.config.get_trn_label_slice_PNG_dir(),
                                     self.config.get_trn_label_slice_NPY_dir(),
                                     thr_files[i],
                                     self.config.get_thr_label_slice_PNG_dir(),
                                     self.config.get_thr_label_slice_NPY_dir())

    def acquire_patches(self, balanced=True):
        """Extract patches from slices of CT scans."""
        print('* Extracting patches from CT-scan slices.')

        # Find all numpy arrays in directories
        trn_CTs_slices = sorted(glob(self.config.get_trn_CT_slice_NPY_dir()
                                     + '*.npy'))
        trn_lbl_slices = sorted(glob(self.config.get_trn_label_slice_NPY_dir()
                                     + '*.npy'))
        thr_lbl_slices = sorted(glob(self.config.get_thr_label_slice_NPY_dir()
                                     + '*.npy'))

        # Call an instance of Scans
        scans = Scans(trn_CTs_slices, trn_lbl_slices, thr_lbl_slices)

        # Return patches sampled from scans
        return scans.sample_patches(classes=self.config.get_classes(),
                                    patch_size=self.config.get_patch_size(),
                                    num_patches=self.config.get_num_patches(),
                                    edges=self.config.get_edges(),
                                    balanced=balanced)

    def initialise_network(self):
        """Initialize a network architecture."""
        print('* Initializing network.')

        # Construct an optimizer
        if self.config.get_optimizer() == 'SGD':
            opt = ks.optimizers.SGD(lr=self.config.get_learning_rate(),
                                    decay=self.config.get_decay(),
                                    momentum=self.config.get_momentum(),
                                    nesterov=self.config.get_Nesterov())

        elif self.config.get_optimizer() == 'RMSprop':
            opt = ks.optimizers.RMSprop(lr=self.config.get_learning_rate(),
                                        rho=self.config.get_rho(),
                                        epsilon=self.config.get_epsilon(),
                                        decay=self.config.get_decay())

        else:
            raise ValueError('Optimizer type not supported.')

        # Initialize model
        self.model = CNN_Model(optimizer=opt,
                               patch_size=self.config.get_patch_size(),
                               num_epochs=self.config.get_num_epochs(),
                               num_classes=len(self.config.get_classes()),
                               batch_size=self.config.get_batch_size(),
                               weight_reg=self.config.get_weight_reg()[0],
                               dropout=self.config.get_dropout(),
                               num_filters=self.config.get_num_filters(),
                               kernel_dims=self.config.get_kernel_dims(),
                               activation=self.config.get_activation())

        self.model.compile_single()

    def cross_evaluation(self, X, Y, Yt, patient_index, num_folds=2,
                         predict_im=False):
        """
        Train and evaluate the model on a patient hold-out basis.

        Call the CNN_Model's cross-validation method and report metrics of
        interest.
        """
        # Start cross-validation procedure
        acc, preds = self.model.cross_validate(X, Y, patient_index,
                                               num_folds=self.config
                                               .get_num_folds())

        # Map label one-hot encoding to label vector
        y = np.argmax(Y, axis=1)
        yt = np.argmax(Yt, axis=1)

        # Report performance of the model
        print('* Classification report CNN:')
        print(classification_report(y, preds))
        print('Accuracy model = ', acc)

        # Report performance of thresholding for comparison
        print('* Classification report for thresholding:')
        print(classification_report(y, yt))
        print('Accuracy thresholding = ', np.mean(y == yt, axis=0))


if __name__ == '__main__':
    """Example usage of Deepy3d class."""

    # Initialize instance of Deepy3d
    deepy3d = Deepy3d('config.yml')

    # Split data up into training, validation and testing
    deepy3d.process_training_data()

    # Extract patches from patient scans
    X, Y, Yt, patient_index = deepy3d.acquire_patches()

    # Initialise the convolutional neural network
    deepy3d.initialise_network()

    # Train and evaluate the model on a hold-out basis
    deepy3d.cross_evaluation(X, Y, Yt, patient_index)

    # Visualise learned kernels
    deepy3d.model.vis_kernels(which_layer=[0, 4, 8, 12], show=False)
