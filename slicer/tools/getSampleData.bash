#!/bin/bash

curDir=$(pwd)
destDir=$curDir/data
mkdir -p $destDir && cd $destDir

echo 'Downloading data to ' $destDir && echo


# Sample registration data
echo 'Downloading registration data' && echo
dataSrc='https://www.slicer.org/w/images/e/eb/RegistrationData.zip'
mkdir -p registration && cd registration
bsdtar -xf <(curl -qL $dataSrc)
rm -rf __MACOSX/
cd ..

# Sample CT data
echo 'Downloading CT chest data' && echo
dataSrc='http://www.slicer.org/slicerWiki/images/3/31/CT-chest.nrrd'
mkdir -p CT && cd CT
curl -qL $dataSrc -o CT-chest.nrrd
cd ..
