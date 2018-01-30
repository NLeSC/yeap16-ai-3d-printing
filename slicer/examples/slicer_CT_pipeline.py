#!/usr/bin/env python

import os
import slicer

numberOfScans = 1

# get current script path
scriptPath = os.path.dirname(os.path.realpath(__file__))
dataPath = scriptPath + '/../data/CT/'

# Create Hierarchy for model making
modelHNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLModelHierarchyNode')
modelHNode.SetName('Skull_cap_models')
modelHNode = slicer.mrmlScene.AddNode(modelHNode)

# Define modelmaker
ModelMaker = slicer.modules.modelmaker

for scan in range(numberOfScans):

    # load CT file of sample CT-shest
    slicer.util.loadVolume(dataPath + 'CT-chest.nrrd')
    ctImage = slicer.util.getNode('CT-chest')

    # use vtk to adjust the threshold value
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(ctImage.GetImageData())
    # TODO: adjust a better threshold level
    threshold.ThresholdBetween(300, 1000)
    threshold.SetInValue(255)
    threshold.SetOutValue(0)

    #  use a slicer serode
    serode = slicer.vtkImageErode()
    serode.SetInputConnection(threshold.GetOutputPort())
    serode.SetNeighborTo4()
    serode.Update()

    ctImage.SetAndObserveImageData(serode.GetOutputDataObject(0))

    # save label map
    outFileName='CT_label_{}.nrrd'.format(scan)
    outFilePath = os.path.join(dataPath, outFileName)
    slicer.util.saveNode(ctImage, outFilePath)

    # make model out of label
    parameters = {}

    parameters['InputVolume'] = ctImage.GetID()
    parameters['ModelSceneFile'] = modelHNode.GetID()
    parameters['Name'] = 'Skull_cap_{}'.format(scan)

    outmodel = slicer.cli.run(ModelMaker,None,parameters,True)
    modelname = 'Skull_cap_{}'.format(scan) + '_255_R_1.000_G_1.000_B_1.000_A_1.000'
    model = slicer.util.getNode(modelname)

    # Save model
    outFileName = 'CT_model_{}.vtk'.format(scan)
    outFilePath = os.path.join(dataPath, outFileName)

    slicer.util.saveNode(model,outFilePath)
