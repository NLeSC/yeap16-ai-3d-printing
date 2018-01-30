#!/bin/sh

cd $(mktemp -d)
tmpDir=$(pwd)
slicerDir=/opt/slicer

curl 'http://slicer.kitware.com/midas3/download?folders=&items=262752-' | tar xz

sudo mv Slicer-* $slicerDir && cd && rm -rf $tmpDir


if [[ :$PATH: == *:"$slicerDir":* ]] ; then
    echo "Found $slicerDir in your PATH"
else
    echo 'export PATH=$PATH:/opt/slicer'  >> ~/.bash_profile
    echo 'export PATH=$PATH:/opt/slicer'  >> ~/.bashrc
fi



