#!/usr/bin/env python

import os
import slicer

# number of scans that have to be converted.
numberOfScans = 1 

# Current directories
scriptPath = os.path.dirname(os.path.realpath(__file__))
dataPath = scriptPath + '/../data/CT/'

# load CT file
for scan in numberOfScans:
    slicer.util.loadVolume(dataPath + 'CT_label{}.nrrd'.format(scan))
