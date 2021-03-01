#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nrrd
import numpy as np
import numpngw as pw
import progressbar as pb
from os.path import basename, isfile
from glob import glob

progress = pb.ProgressBar(widgets=[pb.Bar('*', '[', ']'),
                          pb.Percentage(), ' '])


class PatientIO(object):
    """Process a CT scan into 2D slices and label maps."""

    def __init__(self, filename):
        """
        Instance of a PatientIO class.

        INPUT:  (1) filename: name of CT-scan file
        """
        self.filename = filename
        self.patientID = int(basename(self.filename).split('_')[0])
        self.CT_scan, self.scan_info = self.read_scan(filename)
        self.num_slices = self.CT_scan.shape[-1]

    def read_scan(self, filename):
        """
        Read the CT-scan for further processing.

        Input   (1) str   'filename': name of CT-scan file
        Output  (1) array 'CT_scan': slices by height by width
                (2) array 'scaninfo': header information of the CT-scan file
        """
        CT_scan, scaninfo = nrrd.read(self.filename)

        # Normalize whole scan between 0 and 1
        CT_scan -= np.min(CT_scan)
        CT_scan = CT_scan / np.max(CT_scan).astype('float')

        return CT_scan, scaninfo

    def save_scan(self, patientID, dir_PNG, dir_NPY, overwrite=False):
        """
        Save separate axial CT slices.

        INPUT:  (1) int 'patientID': unique identifier for each patient.
                (2) string 'dir_PNG': directory to save PNG files.
                (3) string 'dir_NPY': directory to save Numpy files
                (4) Boolean 'overwrite': overwrite saved CT NPY and PNG slices
        OUTPUT: saves CT slices
        """
        print(' * patient {}...'.format(self.patientID))
        progress.currval = 0

        for slice_ix in progress(xrange(self.num_slices)):

            # Take out slice
            CT_slice = self.CT_scan[:, :, slice_ix]

            # NPY filename
            npy_fn = dir_NPY + '{}_{}.npy'.format(self.patientID, slice_ix)

            # Check if numpy file exists and whether overwrite is turned on
            if not (isfile(npy_fn) and not overwrite):

                # Save CT_slice as 2D numpy array
                np.save(npy_fn, CT_slice)

            # PNG filename
            png_fn = dir_PNG + '{}_{}.png'.format(self.patientID, slice_ix)

            # Check if PNG file exists and whether overwrite is turned on
            if not (isfile(png_fn) and not overwrite):

                # Rescale slice from [0,1] to [0,...255]
                CT_slice = (255.*CT_slice).astype(np.uint8)

                # Save CT_slice as png
                pw.write_png(png_fn, CT_slice)

    def read_save_labels(self, patientID,
                         label_path,
                         dir_label_PNG,
                         dir_label_NPY,
                         threshold_label_path,
                         dir_threshold_label_PNG,
                         dir_threshold_label_NPY,
                         overwrite=False):
        """
        Read gold standard label maps and saves them as label slices.

        INPUT:  (1) int 'patientID': unique identifier for each patient.
                (2) str 'label_path': filename of the label.
                (3) str 'dir_label_PNG': save directory of PNG labels.
                (4) str 'dir_label_NPY': save directory of Numpy labels.
                (5) str 'threshold_label_path': directory where the threshold
                    labels are currently saved.
                (6) str 'dir_threshold_label_PNG': save directory of the PNG
                    threshold labels.
                (7) str 'dir_threshold_label_NPY': save directory of the Numpy
                    threshold labels.
                (8) Boolean 'overwrite': overwrite saved NPY and PNG labels
        OUTPUT: saves gold standard labels and threshold labels in the given
                directories
        """
        print('Saving labels for patient {}...'.format(self.patientID))
        progress.currval = 0

        # Read the label (segmented CT-scan)
        slices, _ = nrrd.read(label_path)

        # Read the thresholded file
        threshold_slices, _ = nrrd.read(threshold_label_path)

        for slice_idx in range(self.num_slices):

            # Label npy filename
            lab_npy_fn = dir_label_NPY+'{}_{}.npy'.format(self.patientID,
                                                          slice_idx)

            # Check if numpy file exists and whether overwrite is turned on
            if not (isfile(lab_npy_fn) and not overwrite):

                # Write label slice as numpy array
                np.save(lab_npy_fn, slices[:, :, slice_idx].astype('uint8'))

            # Threshold npy filename
            thr_npy_fn = dir_threshold_label_NPY+'{}_{}.npy'.format(
                            self.patientID, slice_idx)

            # Check if numpy file exists and whether overwrite is turned on
            if not (isfile(thr_npy_fn) and not overwrite):

                # Write thresholded label slice as numpy array
                np.save(thr_npy_fn, threshold_slices[:, :, slice_idx]
                        .astype('uint8'))

            # Label png filename
            lab_png_fn = dir_label_PNG+'{}_{}.png'.format(self.patientID,
                                                          slice_idx)

            # Check if PNG file exists and whether overwrite is turned on
            if not (isfile(lab_png_fn) and not overwrite):

                # Write label slice as png
                pw.write_png(lab_png_fn, slices[:, :, slice_idx].astype('uint8'), bitdepth=1)

            # Threshold png filename
            thr_png_fn = dir_threshold_label_PNG+'{}_{}.png'.format(
                            self.patientID, slice_idx)

            # Check if PNG file exists and whether overwrite is turned on
            if not (isfile(thr_png_fn) and not overwrite):

                # Write thresholded label slice as png
                pw.write_png(thr_png_fn, threshold_slices[:, :, slice_idx]
                             .astype('uint8'), bitdepth=1)
