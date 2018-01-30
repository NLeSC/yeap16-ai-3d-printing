#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
from os.path import basename
from itertools import compress
from matplotlib import cm
from matplotlib.patches import Rectangle

progress = pb.ProgressBar(widgets=[pb.Bar('*', '[', ']'),
                          pb.Percentage(), ' '])


class Scans(object):
    """
    Class for manipulating CT-scans of patients.

    Contains functions for extracting and manipulating patches as well as
    perform assertions on slices.
    """

    def __init__(self, scan_files, segmented_files, threshold_files,
                 num_necessary=2e3):
        """
        Initialize an instance of a Scans class.

        INPUT   (1) list 'scan_files': all files to be included for patch
                    selection.
                (2) list 'segmented_files': all labelled files to be included
                    for patch selection.
                (3) list 'threshold_files': all thresholded files to be
                    included for patch selection.
                (4) int 'num_necessary': number of required bone voxels.
        """
        # Remove all patients without sufficient bone voxels
        slices = self.check_bone(segmented_files, num_necessary)

        self.scan_files = list(compress(scan_files, slices))
        self.segmented_files = list(compress(segmented_files, slices))
        self.threshold_files = list(compress(threshold_files, slices))
        self.num_slices = len(self.scan_files)

    def check_bone(self, segmented_files, num_necessary=1):
        """
        Check for sufficient bone voxels.

        INPUT   (1) list ;segmented_files': All segmented files
                (2) int num_necessary: number of bone voxels necessary
        Return  (1) boolean list of the segmented files that indicates whether
                    they contain sufficient bone.
        """
        check = []
        for n, segm in enumerate(segmented_files):

            # Read segmentation file with numpy
            L = np.load(segm)

            # Check number of bone voxels
            check.append((np.sum(L) > num_necessary))

        return check

    def sample_patches(self, classes, patch_size=(3, 3), num_patches=1,
                       edges=(0, 0), balanced=True):
        """
        Sample patches per slice and per tissue.

        INPUT:  (1) list 'classes': which classes to sample from.
                (2) tuple 'patch_size': height and width of patch.
                (3) int 'num_patches': number of patches to sample.
                (4) tuple 'edges': size of edges not be sampled from.
                (5) boolean 'balanced': whether to use balanced sampling
        OUTPUT: (1) array 'patches': num_patches by patch_size
                (2) array 'labels': num_patches by num_classes
                (3) array 'threshold': num_patches by num_classes
                (4) array 'patient_index': num_patches by 1
        """
        # Number of classes
        nK = len(classes)

        # Patch height and width
        ph = patch_size[0]
        pw = patch_size[1]

        # Patch deviation in each direction
        patch_dev = ((ph-1)/2, (pw-1)/2)

        # Initialize patches and labels
        labels = np.zeros((num_patches, nK), dtype='uint8')
        patches = np.zeros((num_patches, ph, pw, 1))
        threshold_labels = np.zeros((num_patches, nK), dtype='uint8')
        patient_index = np.zeros(num_patches, )

        if balanced:
            # Number of patches to draw from each patient and each class
            num_draw = num_patches / self.num_slices / nK

            # Check whether num_draw is larger than 1
            if num_draw < 1:
                raise ValueError('''Number of patches to draw is smaller than
                                 the number of slices * number of tissues''')

            # Initialize counter
            ct = 0

            # Iterate over classes to sample from
            for n in range(self.num_slices):

                # Read scan from n-th patient
                scan_path = self.scan_files[n]
                scan = np.load(scan_path)

                # Read the corresponding label
                segmented_path = self.segmented_files[n]
                segm = np.load(segmented_path)

                # Read the corresponding threshold label
                threshold_path = self.threshold_files[n]
                threshold = np.load(threshold_path)

                # Find region of interest.
                upperEdge = np.min(np.argwhere(segm == 1), 0)[0]
                lowerEdge = np.max(np.argwhere(segm == 1), 0)[0]+1
                rightEdge = np.max(np.argwhere(segm == 1), 0)[1]+1
                leftEdge = np.min(np.argwhere(segm == 1), 0)[1]

                # re-label region out of interest
                segm[lowerEdge:] = np.max(classes)+1
                segm[0:upperEdge] = np.max(classes)+1
                segm[:, rightEdge:] = np.max(classes)+1
                segm[:, 0:leftEdge] = np.max(classes)+1

                for k, cl in enumerate(classes):

                    # Find indices of current class in segmented image
                    ix = np.argwhere(segm[edges[0]+patch_dev[0]:
                                          -edges[0]-patch_dev[0]-1,
                                          edges[1]+patch_dev[1]:
                                          -edges[1]-patch_dev[1]-1] == cl)

                    # Add offset by edges and patch center-edge distance
                    ix = ix + edges + patch_dev

                    # Throw error of slice does not contain current tissue
                    if ix.shape[0] <= 0:
                        raise ValueError('Slice does not contain tissue' +
                                         str(k))

                    # Choose random subset of k-th class indices
                    ix = ix[np.random.choice(ix.shape[0], size=num_draw), :]

                    # Extract patch
                    for i in range(num_draw):

                        # Slice patch
                        patches[ct, :, :, 0] = scan[ix[i, 0] - patch_dev[0]:
                                                    ix[i, 0] + patch_dev[0]+1,
                                                    ix[i, 1] - patch_dev[1]:
                                                    ix[i, 1] + patch_dev[1]+1]

                        # Include label
                        labels[ct, k] = 1

                        # Include threshold label
                        threshold_labels[ct, threshold[ix[i, 0], ix[i, 1]]] = 1

                        # Include patient index
                        patient_index[ct] = int(basename(scan_path)
                                                .split('_', 1)[0])

                        # print progress
                        if np.mod(ct, num_patches/10) == 0:
                            print(round(ct*100/num_patches),
                                  '% of the patches has been created')

                        # Increment counter
                        ct += 1

        else:
            # Number of patches to draw from each patient
            num_draw = num_patches / self.num_slices

            # Check whether num_draw is larger than 1
            if num_draw < 1:
                raise ValueError('''Number of patches to draw is smaller than
                                 the number of slices * number of tissues''')

            # Initialize counter
            ct = 0

            for n in range(self.num_slices):

                # Read scan from n-th patient
                scan_path = self.scan_files[n]
                scan = np.load(scan_path)

                # Read the corresponding label
                segmented_path = self.segmented_files[n]
                segm = np.load(segmented_path)

                # Read the corresponding threshold label
                threshold_path = self.threshold_files[n]
                threshold = np.load(threshold_path)

                # Image shape
                h, w = scan.shape

                # Find region of interest.
                upperEdge = np.max(np.array([np.min(np.argwhere(segm == 1), 0)
                                            [0], patch_dev[0]]))
                lowerEdge = np.min(np.array([np.max(np.argwhere(segm == 1), 0)
                                            [0], w-patch_dev[0]]))
                rightEdge = np.min(np.array([np.max(np.argwhere(segm == 1), 0)
                                            [1], h-patch_dev[1]]))
                leftEdge = np.max(np.array([np.min(np.argwhere(segm == 1), 0)
                                            [1], patch_dev[1]]))

                # Meshgrid of image wihout edges and patch deviations
                ixh, ixw = np.meshgrid(range(upperEdge, lowerEdge+1),
                                       range(leftEdge, rightEdge+1))

                # Choose random subset of k-th class indices
                ix = np.random.choice(ixh.size, size=num_draw)
                ixh = ixh.ravel()[ix]
                ixw = ixw.ravel()[ix]

                # Extract patch
                for i in range(num_draw):

                    # Slice patch
                    patches[ct, :, :, 0] = scan[ixh[i] - patch_dev[0]:
                                                ixh[i] + patch_dev[0]+1,
                                                ixw[i] - patch_dev[1]:
                                                ixw[i] + patch_dev[1]+1]

                    # Store segmentation labels
                    labels[ct, segm[ixh[i], ixw[i]]] = 1

                    # Store threshold labels
                    threshold_labels[ct, threshold[ixh[i], ixw[i]]] = 1

                    # Find current patient index
                    patient_index[ct] = int(basename(scan_path)
                                            .split('_', 1)[0])

                    # Increment counter
                    ct += 1

        # Remove excess pre-allocated entries in arrays
        labels = labels[0:ct-1]
        patches = patches[0:ct-1]
        threshold_labels = threshold_labels[0:ct-1]
        patient_index = patient_index[0:ct-1]

        return patches, labels, threshold_labels, patient_index

    def subpatches(self, X, patch_size):
        """Draw smaller patches from given patches."""
        # Patch deviation in each direction
        pd = ((patch_size[0]-1)/2, (patch_size[1]-1)/2)

        # Center pixel
        center = ((X.shape[1]-1)/2, (X.shape[2]-1)/2)

        # Return sliced array
        return X[:, center[0]-pd[0]:center[0]+pd[0]+1,
                 center[1]-pd[1]:center[1]+pd[1]+1, :]
