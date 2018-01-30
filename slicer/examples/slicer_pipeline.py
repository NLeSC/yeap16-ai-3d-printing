#!/usr/bin/env python

import os


def loadVolume(dataPath, scanNumber):
    '''
    Loads a CT file
    '''
    fileList = os.listdir(dataPath)
    slicer.util.loadVolume(dataPath + fileList[scanNumber])

    volume = slicer.util.getNode(os.path.splitext(fileList[scanNumber])[0])
    return volume


def createHierarchy(): 
    '''
    Create Hierarchy for model making
    '''
    modelHNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLModelHierarchyNode')
    modelHNode.SetName('Models')
    slicer.mrmlScene.AddNode(modelHNode)

    return modelHNode


def thresholdEffect(inputVolume, lowerBound, upperBound): # Use vtk to adjust threshold

    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(inputVolume.GetImageData())
    threshold.ThresholdBetween(lowerBound, upperBound)
    threshold.SetInValue(255)
    threshold.SetOutValue(0)

    #  use a slicer serode
    serode = slicer.vtkImageErode()
    serode.SetInputConnection(threshold.GetOutputPort())
    serode.SetNeighborTo4()
    serode.Update()

    inputVolume.SetAndObserveImageData(serode.GetOutputDataObject(0))

    return inputVolume


def modelMaker(thresholdVolume, hierarchyNode, scanNumber):
    '''
    Defines a model
    '''
    # Define parameters for model making
    parameters = {}
    parameters['InputVolume'] = thresholdVolume.GetID()
    parameters['ModelSceneFile'] = hierarchyNode.GetID()
    parameters['Name'] = 'Skull_cap_{}'.format(scanNumber)

    # Create model
    modelMaker = slicer.modules.modelmaker
    slicer.cli.run(modelMaker, None, parameters, True)
    modelName = parameters['Name'] + '_255_R_1.000_G_1.000_B_1.000_A_1.000' # Standard extension with label number and color values.
    model = slicer.util.getNode(modelName)

    return model


def saveFile(file, outFileName, outFilePath):
    '''
    Saves the node in a file
    '''

    #FileName = outFileName + '_{}'.format(scan_num)
    dataPath = os.path.join(outFilePath, outFileName)
    slicer.util.saveNode(file, dataPath)


if __name__ == "__main__":

    scanNumber = 1
    scriptPath = os.path.dirname(os.path.realpath(__file__))
    dataPath = scriptPath + '/../data/CT/'
    saveLabelToPath = scriptPath + '/../data/Labels/'
    saveModelToPath = scriptPath + '/../data/Models/'

    modelHNode = createHierarchy()

    for scan in range(scanNumber):

        volume = loadVolume(dataPath, scan)
        thresholdVolume = thresholdEffect(volume,300,3000)
        saveFile(thresholdVolume,'CT_label_{}.mha'.format(scan), saveLabelToPath)
        model = modelMaker(thresholdVolume, modelHNode, scan)
        saveFile(model, 'CT_model_{}.vtk'.format(scan), saveModelToPath)

        
