import os

# get current script path
scriptPath = os.path.dirname(os.path.realpath(__file__))
dataPath = scriptPath + '/../data/CT/'

# load CT file
slicer.util.loadVolume(dataPath + 'CT-chest.nrrd')
#print slicer.util.getNodes()
chest = slicer.util.getNode('CT-chest')

# use vtk to adjust the threshold value
threshold = vtk.vtkImageThreshold()
threshold.SetInputData(chest.GetImageData())
threshold.ThresholdBetween(100, 200)
threshold.SetInValue(255)
threshold.SetOutValue(0)


#  use a slicer serode
serode = slicer.vtkImageErode()
serode.SetInputConnection(threshold.GetOutputPort())
serode.SetNeighborTo4()
serode.Update()

chest.SetAndObserveImageData(serode.GetOutputDataObject(0))

outFileName='CT_chest_out.nrrd'
outFilePath = os.path.join(dataPath, outFileName)
slicer.util.saveNode(chest, outFilePath)


exit()
