#!/bin/bash

# Arguments: 
#   mesh filename (mesh.vtk)
#   reference image
#   output(labelmap) name


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/libs

./bin/MeshToLabelMap \
    --input_mesh $1 \
    --reference_volume $2 \
    --output_labelmap $3 \
    --pixel_value 1 \
    --spacing 1,1,1 \
    --boundary_extension 1,1,1 \
    --median_radius -1,-1,-1 \
    --verbose

