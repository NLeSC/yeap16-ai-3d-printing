#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
from glob import glob


class ConfigReader(object):
    """
    Class to read YAML configuation files.

    Provided methods are for retrieving and reporting the properties of
    a given YAML configuration file.
    """
    def __init__(self, config_file):
        """Initialize ConfigReader with a yaml config file."""
        self.config_file = config_file
        self.read_config(config_file)

    def read_config(self, config_file):
        """Read yaml config file."""
        with open(config_file, 'r') as ymlfile:
            self.config = yaml.load(ymlfile)

    def get_config(self):
        """Return the config file itself."""
        return self.config

    def show_config(self):
        """Print sections of config file."""
        for section in self.config:
            print(section)

    def get_section(self, sectionName):
        """Retrieve a particular section of the config file."""
        if sectionName in self.config:
            return self.config[sectionName]
        else:
            print('Section does not exist.')
            return -1

    '''File directories for training, testing and comparing models.'''

    def get_base_dir(self):
        """Retrieve root file directory."""
        return self.config['paths']['base_dir']

    def get_trn_CT_dir(self):
        """Retrieve directory containing CT-scans for training."""
        return self.get_base_dir() + self.config['paths']['trn_CT_dir']

    def get_trn_CT_slice_PNG_dir(self):
        """Retrieve directory of CT-slices for training in PNG format."""
        return self.get_base_dir() + \
            self.config['paths']['trn_CT_slice_PNG_dir']

    def get_trn_CT_slice_NPY_dir(self):
        """Retrieve directory of CT-slices for training in NPY format."""
        return self.get_base_dir() + \
            self.config['paths']['trn_CT_slice_NPY_dir']

    def get_trn_label_dir(self):
        """Retrieve directory of label files for training."""
        return self.get_base_dir() + self.config['paths']['trn_label_dir']

    def get_trn_label_slice_PNG_dir(self):
        """Retrieve directory of label files for training in PNG format."""
        return self.get_base_dir() + \
            self.config['paths']['trn_label_slice_PNG_dir']

    def get_trn_label_slice_NPY_dir(self):
        """Retrieve directory of  label files for training in NPY format."""
        return self.get_base_dir() + \
            self.config['paths']['trn_label_slice_NPY_dir']

    def get_thr_label_dir(self):
        """Retrieve directory of threshold-label files."""
        return self.get_base_dir() + self.config['paths']['thr_label_dir']

    def get_thr_label_slice_PNG_dir(self):
        """Retrieve directory of threshold-label files in NPY format."""
        return self.get_base_dir() + \
            self.config['paths']['thr_label_slice_PNG_dir']

    def get_thr_label_slice_NPY_dir(self):
        """Retrieve directory of threshold-label files in NPY format."""
        return self.get_base_dir() + \
            self.config['paths']['thr_label_slice_NPY_dir']

    def get_trn_CT_patch_dir(self):
        """Retrieve directory of patches from CT-slices for training."""
        return self.get_base_dir() + self.config['paths']['trn_CT_patch_dir']

    def get_trn_label_patch_dir(self):
        """Retrieve directory of patches from labels for training."""
        return self.get_base_dir() + \
            self.config['paths']['trn_label_patch_dir']

    def get_pred_dir(self):
        """Retrieve directory of predictions made by system."""
        return self.get_base_dir() + self.config['paths']['pred_label_dir']

    def get_CT_scans(self):
        """Retrieve CT-scan filenames."""
        return sorted(glob(self.get_trn_CT_dir() + '*' +
                      self.config['training']['file_extension']))

    def get_CT_labels(self):
        """Retrieve CT-scan segmentation filenames."""
        return glob(self.get_trn_label_dir() + '*')

    def get_CT_thresholded(self):
        """Retrieve CT-scan threshold-segmentation filenames."""
        return glob(self.get_thr_label_dir() + '*')

    def get_tst_images(self):
        """Retrieve images for testing."""
        return glob(self.get_base_dir() +
                    self.config['paths']['test_image_dir'] + '*')

    def get_tst_labels(self):
        """Retrieve label images for testing."""
        return glob(self.get_base_dir() +
                    self.config['paths']['test_label_dir'] + '*')

    '''Properties of data of patients.'''

    def get_num_patients(self):
        """Return number of patients to use."""
        return self.config['patient_data']['num_patients']

    def get_classes(self):
        """Return list containing which classes are to be segmented."""
        return self.config['patient_data']['classes']

    '''Properties of patches extracted from CT-scan slices.'''

    def get_patch_size(self):
        """Return tuple with dimensions of patch."""
        return (self.config['patches']['size']['x'],
                self.config['patches']['size']['y'])

    def get_edges(self):
        """Return tuple with sizes of image edges."""
        return (self.config['patches']['edges']['x'],
                self.config['patches']['edges']['y'])

    def get_num_patches(self):
        """Return number of patches."""
        return self.config['patches']['num_patches']

    def get_patch_save_dir(self):
        """Retrieve save directory of patches"""
        return self.config['patches']['save_dir']

    '''Network configuration and training settings.'''

    def get_trn_type(self):
        """Retrieve type of training."""
        return self.config['training']['type']

    def get_architecture(self):
        """Retrieve architecture for training."""
        return self.config['training']['architecture']

    def get_val_split(self):
        """Return percentage of data to be held back for validation."""
        return self.config['training']['val_split']

    def get_num_epochs(self):
        """Return number of epochs that the network should be trained for."""
        return self.config['training']['num_epochs']

    def get_num_folds(self):
        """Return number of folds for cross-evaluation"""
        return self.config['training']['num_folds']

    def get_batch_size(self):
        """Return the number of samples in a single batch."""
        return self.config['training']['batch_size']

    def get_weight_reg(self):
        """Return the regularization parameter for the weights."""
        return self.config['training']['weight_reg']

    def get_dropout(self):
        """Retrieve the value of the dropout parameter."""
        return self.config['training']['dropout']

    def get_num_filters(self):
        """Return the list with the number of filters in each layer."""
        return self.config['training']['num_filters']

    def get_kernel_dims(self):
        """Return the list with the dimensions of the filters in each layer."""
        return self.config['training']['kernel_dims']

    def get_activation(self):
        """Retrieve the nonlinear activation function."""
        return self.config['training']['activation']

    '''Optimization algorithm properties.'''

    def get_optimizer(self):
        """Return which optimization algorithm to use."""
        return self.config['optimizer']['type']

    def get_learning_rate(self):
        """Return learning rate for the gradient."""
        return self.config['optimizer']['learning_rate']

    def get_decay(self):
        """Return parameter for the decay of the learning rate."""
        return self.config['optimizer']['decay']

    def get_momentum(self):
        """Return momentum parameter."""
        return self.config['optimizer']['momentum']

    def get_Nesterov(self):
        """Retrieve parameter for Nesterov momentum."""
        return self.config['optimizer']['Nesterov']

    def get_rho(self):
        """Return decay rate for accumulated gradient."""
        return self.config['optimizer']['rho']

    def get_epsilon(self):
        """Return convergence tolerance parameter."""
        return self.config['optimizer']['epsilon']

    '''Model evaluation settings.'''

    def get_num_tst_patches(self):
        """Return number of patches for evaluation."""
        return self.config['evaluation']['num_tst_patches']

    def get_num_tst_folds(self):
        """Return number of folds for evaluation."""
        return self.config['evaluation']['ntestFolds']

    def get_results_save_dir(self):
        """Retrieve directory for writing results."""
        return self.get_base_dir() + self.config['evaluation']['save_dir']

    '''Model load and save settings.'''

    def get_load_model_name(self):
        """Retrieve name of model to be loaded."""
        return self.config['model']['load_model_name']

    def get_model_weights_save_dir(self):
        """Retrieve directory of where model weights should be stored."""
        return self.get_base_dir() + self.config['model']['weights_save_dir']

    def get_model_architecture_save_dir(self):
        """Retrieve directory of where model architecture should be stored."""
        return self.get_base_dir() + \
            self.config['model']['architecture_save_dir']
