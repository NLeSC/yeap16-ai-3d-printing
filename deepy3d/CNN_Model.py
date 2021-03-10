#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import nrrd
import six
import numpy as np
import numpy.random as rnd
import numpngw as pw
import matplotlib.pyplot as plt
import keras as ks
import keras.backend as K
from glob import glob
from os.path import basename
from scipy.signal import convolve2d
from matplotlib.image import imread, imsave
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1_l2
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.initializers import RandomNormal, RandomUniform

from Scans import Scans
from util import get_closest_factors, block_index


class CNN_Model(object):
    """
    Class of CNN models with training, validation, testing, reporting
    and visualising methods.
    """
    def __init__(self, optimizer='rmsprop', patch_size=(33, 33), num_epochs=8,
                 num_classes=2, batch_size=32, weight_reg=0.01, dropout=0.1,
                 num_filters=[20, 20, 20, 20], kernel_dims=[7, 5, 5, 3],
                 activation='relu'):
        """
        Initialize an instance of a CNN model.
        INPUT   (1) str 'optimizer': optimization algorithm (choices:
                    'rmsprop', 'sgd', or self-defined opt).
                (2) tuple 'patch_size': x and y dimension of the input patches
                (3) int 'num_epochs': number of training epochs (def: 8).
                (4) int 'num_classes': number of tissue classes (def: 2).
                (5) int 'batch_size': size of batch number of images to train
                    on for each batch (def:32).
                (6) float 'weight_reg': l2 regularization parameter(def: 0.01)
                (7) float dropout: proportion of samples to drop out (def: 0.1)
                (7) list 'num_filters': number of filters for each
                    convolutional layer (def: [20,20,20,20])
                (8) list 'kernel_dims': dimension of kernel at each layer
                    (def: [7,5,5,3])
                (9) string 'activation': nonlinear activation function to use
                    at each convolutional layer (def: relu)
        """
        self.optimizer = optimizer
        self.patch_size = patch_size
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.weight_reg = weight_reg
        self.dropout = dropout
        self.num_filters = num_filters
        self.kernel_dims = kernel_dims
        self.activation = activation
        self.net = []

    def compile_single(self):
        """
        Compiles standard single-path model with 4 convolutional /
        max-pooling layers.

        INPUT   None
        OUTPUT  None
        """
        print '    * Compiling single model...'

        # Configure network according to a sequential architecture
        single = Sequential()

        # 1th layer block
        single.add(Conv2D(self.num_filters[0], (self.kernel_dims[0],
                          self.kernel_dims[0]), activation=self.activation,
                          activity_regularizer=l2(self.weight_reg),
                          kernel_initializer=RandomUniform(minval=-0.05,
                          maxval=0.05), bias_initializer='zeros',
                          input_shape=(self.patch_size[0], self.patch_size[1],
                          1), padding="valid", name="conv2d_1"))
        single.add(BatchNormalization(axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        single.add(Dropout(self.dropout))

        # 2nd layer block
        single.add(Conv2D(self.num_filters[1], (self.kernel_dims[1],
                          self.kernel_dims[1]), activation=self.activation,
                          activity_regularizer=l2(self.weight_reg),
                          kernel_initializer=RandomUniform(minval=-0.05,
                          maxval=0.05), bias_initializer='zeros',
                          padding="valid", name="conv2d_2"))
        single.add(BatchNormalization(axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        single.add(Dropout(self.dropout))

        # 3d layer block
        single.add(Conv2D(self.num_filters[2], (self.kernel_dims[2],
                          self.kernel_dims[2]), activation=self.activation,
                          activity_regularizer=l2(self.weight_reg),
                          kernel_initializer=RandomUniform(minval=-0.05,
                          maxval=0.05), bias_initializer='zeros',
                          padding="valid", name="conv2d_3"))
        single.add(BatchNormalization(axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        single.add(Dropout(self.dropout))

        # 4th layer block
        single.add(Conv2D(self.num_filters[3], (self.kernel_dims[3],
                          self.kernel_dims[3]), activation=self.activation,
                          kernel_initializer=RandomUniform(minval=-0.05,
                          maxval=0.05), bias_initializer='zeros',
                          padding="valid", name="conv2d_4"))
        single.add(BatchNormalization(axis=1))
        single.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        # Flatten convolutional activations into vector and map to dense layer.
        single.add(Flatten())
        single.add(Dense(self.num_classes, activation='softmax'))

        # Compile model
        single.compile(loss='categorical_crossentropy',
                       optimizer=self.optimizer, metrics=["accuracy"])

        # Report network architecture
        single.summary()
        print('* Done compiling network architecture.')

        # Store network
        self.net = single

    def fit_model(self, X, Y, val_split=0.2, num_epochs=1):
        """
        INPUT   (1) numpy array 'X': list of patches for training (num_samples,
                    h, w, 1)
                (2) numpy array 'Y'': list of labels for each patch in X
                    (num_samples, nK)
                (3) float 'val_split'= proportion of samples to be set apart
                    for validation during training (def: 0.2).
                (4) int 'num_epochs': number of training epochs.
        OUTPUT  None
        """

        # Shuffle data
        shuffle = zip(X, Y)
        rnd.shuffle(shuffle)
        X = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        Y = np.array([shuffle[i][1] for i in xrange(len(shuffle))])

        # Fit model
        self.net.fit(X, Y, batch_size=self.batch_size, epochs=num_epochs,
                     validation_split=val_split, verbose=1, shuffle=True)

    def cross_validate(self, X, Y, patient_index, num_folds=3, val_split=0.2):
        """
        Compute cross-validated accuracy on training set.

        INPUT   (1) array 'X': patches of CT-scans
                (2) array 'Y': corresponding labels of patches
                (3) list 'patient_index': Patient indices of the patches
                (4) int 'num_folds': number of folds (def: 2)
                (5) float 'val_split': proportion of data to set outside for
                    validation.
        OUTPUT  (1) acc: accuracy on the held-out validation patches
                (2) preds: predictions on the held-out validation patches
        """
        print('* Starting cross-validation')

        # Data shapes
        num_patches, h, w, _ = X.shape

        # Number of patients
        num_patients = len(np.unique(patient_index))

        # Assign each patient to 1 fold
        folds = block_index(num_folds, num_patches)

        # Preallocate prediction array
        preds = np.zeros((num_patches,), dtype='uint8')

        # Loop over folds
        for f in range(num_folds):

            # Report that training is starting
            print('*Training network, fold ' + str(f+1) + '/' + str(num_folds))

            # Compile network architecture
            self.net.compile(loss='categorical_crossentropy',
                             optimizer=self.optimizer,
                             metrics=["accuracy"])

            # Train model on X,Y
            self.fit_model(X[folds != f, :, :, :], Y[folds != f, :],
                           num_epochs=self.num_epochs, val_split=val_split)

            # Report that predicting is starting
            print('* Predicting')

            # Propagate held-out patches through network
            net_out = self.net.predict(X[folds == f, :])

            # Make predictions on patches from the held out fold
            preds[folds == f] = np.argmax(net_out, axis=1)

        # Compute classification accuracy
        acc = np.mean(preds == np.argmax(Y, axis=1), axis=0)

        # return accuracy on the predicted patches, return predictions
        return acc, preds

    def predict_image(self, test_img, pred_dir, patch_size=(33, 33),
                      num_batches=4, save_segm=False):
        """
        Predict tissues for new image.

        INPUT   (1) str 'test_img': filepath to image to predict.
                (2) str 'pred_dir': Save directory of predictions.
                (3) tuple 'patch_size': patch dimensions (def: (33, 33))
                (4) int 'num_batches': cut patches array into batches (def: 4)
                (5) boolean 'save_segm': save segmented image (def: False)
        OUTPUT  (1) array 'segm': Segmented image
        """

        # Get patch length from center
        patch_step = ((patch_size[0]-1)/2, (patch_size[1]-1)/2)

        # Read test image
        imgs = np.load(test_img)

        # Extract name of image
        img_name = basename(test_img).split('.')[0]

        # Minimum intensity value of image
        edge_value = np.min(imgs)

        print ('The image to be predicted is of size', np.shape(imgs))

        #  Padding with the same value as the values at the edge
        imgs = np.pad(imgs, ((patch_step[0], patch_step[0]),
                             (patch_step[1], patch_step[1])),
                      mode='constant', constant_values=edge_value)

        # Find image height and width
        im_h = np.shape(imgs)[0] - (patch_size[0] - 1)
        im_w = np.shape(imgs)[1] - (patch_size[1] - 1)

        # Find all possible patches in the CT slice
        patches = extract_patches_2d(imgs, patch_size)
        patches = np.expand_dims(patches, axis=4)
        num_patches = len(patches)

        # Split test patches into batches to prevent exceeding GPU memory
        batch_ix = block_index(num_batches, num_patches)

        # Predict the batches of patches based on trained model seperately
        pred = np.zeros((num_patches,))
        for b in range(num_batches):
            pred[batch_ix == b] = np.argmax(self.net.predict_on_batch(
                                            patches[batch_ix == b, :, :, :]),
                                            axis=1)

        # Reshape and save prediction
        segm = pred.reshape((im_h, im_w)).astype('uint8')

        # Save predicted image
        if save_segm:
            savefn = pred_dir + 'Slices/' + img_name + '.png'
            pw.write_png(savefn, segm, bitdepth=1)

        # Report progress
        print('Slice', img_name, 'has been segmented')

        return segm

    def vis_kernels(self, which_layer=[0], vis_last=False,
                    save_figname='trained', show=False):
        """
        Visualize kernels within a layer of a model
        INPUT:  (1) list 'which_layer': selects which layer should be
                    visualized (def: 0)
                (2) bool 'vis_last': whether to visualize last layer
                    (def:False)
                (3) str 'save_figname': name of the figure to be saved
                    (def:'trained')
                (4) bool 'show': whether to directly show the kernels
                    (def:False)
        OUTPUT: None
        """

        # Loop over layers
        for layer in which_layer:

            # Extract weights of layer
            weights, bias = self.net.layers[layer].get_weights()

            # Number of kernels
            num_kernels = weights.shape[3]

            # Initialize subplots
            n_rows, n_cols = get_closest_factors(num_kernels)
            fig = plt.figure(figsize=(20, 20))

            # Loop over kernels
            for k in range(num_kernels):
                ax = fig.add_subplot(n_rows, n_cols, k+1)
                shown_im = ax.imshow(weights[:, :, 0, k] + bias[k],
                                     interpolation=None, cmap='gray')
                ax.set_title('Kernel '+str(k+1))
                ax.axis('off')

                # Show colorbar that matches axes
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(shown_im, cax=cax)

            # Save image to filename, if provided
            if save_figname:
                fig.savefig(save_figname+'_kernels_layer'+str(layer)+'.png',
                            bbox_inches='tight', pad_inches=0)

            # Show image on request
            if show:
                plt.show()

        # Show the last layer (fully-connected)
        if vis_last:

            # Extract weights of layer
            weights, bias = self.net.layers[-1].get_weights()

            # Initialize figure
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

            # Plot weights
            ax.plot(weights)

            # Save image to filename, if provided
            if save_figname:
                fig.savefig(save_figname + '_weights_fclayer' + str(layer) +
                            '.png', bbox_inches='tight', pad_inches=0)

            # Show image on request
            if show:
                plt.show()

    def vis_gradients(self, which_layer=[0], save_figname='trained',
                      show=False):
        """
        Visualize gradients of kernels.

        INPUT:  (1) which_layer: selection of layers to visualize (def: 0)
                (2) str 'save_figname': name of the figure to be saved
                    (def:'trained')
                (3) bool 'show': whether to directly show the kernels
                    (def:False)
        OUTPUT: None
        """

        # Loop over layers
        for layer in which_layer:

            # Obtain output from desired layer
            output = self.net.layers[layer].output

            # Get list of weights
            weights = self.net.layers[layer].trainable_weights

            # Let Keras compute gradients
            gradients = self.net.optimizer.get_gradients(self.net.total_loss,
                                                         weights)

            # Define gradient function
            input_tensors = [self.net.inputs[0], self.net.sample_weights[0],
                             self.net.targets[0], K.learning_phase()]
            obtain_gradients = K.function(inputs=input_tensors,
                                          outputs=gradients)

            # Define random image batch
            X = rnd.random((8, self.patch_size[0], self.patch_size[1], 1))

            # Gradient of model w.r.t. random image batch
            dX, db = obtain_gradients(inputs=[X, [1], [[1]], 0])

            # Number of kernels
            num_kernels = dX.shape[3]

            # Initialize subplots
            n_rows, n_cols = get_closest_factors(num_kernels)
            fig = plt.figure(figsize=(20, 20))

            # Loop over kernels
            for k in range(num_kernels):
                ax = fig.add_subplot(n_rows, n_cols, k+1)
                shown_im = ax.imshow(dX[:, :, 0, k] + db[k],
                                     interpolation=None, cmap='gray')
                ax.set_title('Kernel ' + str(k+1))
                ax.axis('off')

                # Show colorbar that matches axes
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(shown_im, cax=cax)

            # Save image to filename, if provided
            if save_figname:
                fig.savefig(save_figname + '_gradients_layer' + str(layer) +
                            '.png', bbox_inches='tight', pad_inches=0)

            # Show image on request
            if show:
                plt.show()

    def vis_activations(self, image, which_layer=[0], save_figname='trained',
                        num_batches=4, show=False):
        """
        Visualize kernels within a layer of a model.

        INPUT:  (1) array/str 'image': either an image array or a filename from
                    which an image can be read
                (2) list 'which_layer': select layer(s) to visualize (def: 0)
                (3) str 'save_figname': name of the figure to be saved
                (4) int 'num_batches': number of batches for network (def:4)
                (5) bool 'show': whether or not to directly show the kernels
        OUTPUT: None
        """

        # If image is a filename, read it
        if isinstance(image, six.string_types):

            # Read file
            image = imread(image)

        # Normalize slice between 0 and 1
        image -= np.min(image)
        image = image / np.max(image).astype('float')

        # Image shape
        h, w = image.shape

        # Length of step in patch away from the center pixel
        pst = ((self.patch_size[0]-1)/2, (self.patch_size[1]-1)/2)

        # Extract all patches from image
        patches = extract_patches_2d(image, patch_size=self.patch_size)

        # Get number of patches
        num_patches = patches.shape[0]

        # Augment patches array to have 1 channel
        patches = np.expand_dims(patches, axis=4)

        # Cut patches into block index array
        batch_ix = block_index(num_batches, num_patches)

        # Loop over layers
        for layer in which_layer:

            # Define intermediate model
            layer_output = K.function([self.net.layers[0].input,
                                       K.learning_phase()],
                                      [self.net.layers[layer].output])

            # Layer output shape list
            _, act_h, act_w, num_kernels = self.net.layers[layer].output_shape

            # Split patches into batches for feeding through network
            act = np.zeros((num_patches, act_h, act_w, num_kernels))

            # Loop over batches
            for b in range(num_batches):
                # Propagate batch through network
                act[batch_ix == b] = layer_output([patches[batch_ix == b, :],
                                                   0])[0]

            # Loop over kernels
            for k in range(num_kernels):

                # Find patch center indices
                pce = [(act_h - 1)/2 + 1, (act_w - 1)/2 + 1]

                # Reshape activation into image
                act_image = np.reshape(act[:, pce[0], pce[1], k],
                                       [h - pst[0]*2, w - pst[1]*2])

                # Initialize a figure
                fig, ax = plt.subplots(figsize=(10, 10))

                # Show activation image
                im = ax.imshow(act_image, interpolation=None, cmap='gray')

                # Show title
                ax.set_title('Activation of layer=' + str(layer) +
                             ', kernel=' + str(k+1))

                # Show colorbar that matches axes
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax)

                # Save image to filename, if provided
                if save_figname:
                    fig.savefig(save_figname + '_activations_layer' +
                                str(layer) + '_kernel_{}'.format(k) + '.png',
                                bbox_inches='tight', pad_inches=0)

                # Show image on request
                if show:
                    plt.show()
