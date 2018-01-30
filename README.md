# yeap16: CT-image bone segmentation
Code repository of the [Young eScientist Award Project](https://www.esciencecenter.nl/redactional/young-escientist-2016) with the 3D Innovation Lab at the VUmc. This code accompanies a paper titled: <br>

"CT image segmentation of bone for medical additive manufacturing using convolutional neural networks" <br>

which is currently under review.

#### Purpose
Bone segmentation of CT scans is an essential step during medical treatment planning.
The exact thickness, orientation and position of bony structures is required to make patient-specific constructs such as surgical guides and implants. During bone segmentation each pixel in a medical image is classified as either 'bone' or 'background'. Unfortunately, current algorithms either lack robustness and reliability, or require tedious manual interaction ([Van Eijnatten et al., 2018](http://dx.doi.org/10.1016/j.medengphy.2017.10.008)). Therefore, this repository contains a fully-automated convolutional neural network (CNN) to perform bone segmentation of CT scans.

#### Model training
The CNN was trained using CT scans of three patients that were previously treated at the Vrije Universiteit Medical Center. Each pixel of the CT scans was labelled as either "bone" or "background" based on the knowledge of highly experienced medical engineers. A number of 500,000 pixels was randomly selected to create a axial patch of 33x33 around those selected pixels. These patches were subsequently used to train the CNN, which allow the CNN to learn the classification of pixels based on the local environment. Furthermore, a balanced training method was applied as proposed by [Havaei et al. (2016)](https://doi.org/10.1016/j.media.2016.05.004), which implies that "bone" and "background" patches were equally represented in the dataset.

#### CNN architecture
Our CNN architecture was inspired by the one described in the [Github repository](https://github.com/naldeborgh7575/brain_segmentation.git) of N.Aldenborgh. This architecture consists of four convolutional layers with additional ReLU, batch normalization and pooling layers. Pixel classification was performed using the softmax cost function. Furthermore, the RMSprop optimizer was used during the CNN training.  

## Installation
Install dependencies (Ubuntu):
```shell
sudo apt-get install git numpy scipy matplotlib skimage sklearn tensorflow keras numpngw six json nrrd yaml
```
Clone the repository:
```shell
git clone https://github.com/NLeSC/yeap16-ai-3d-printing.git
```
Import Deepy3d class in a python script:
```shell
from Deepy3d import Deepy3d
```

## Usage
When called directly, the Deepy3d class runs an example script that reads in patient data according to a configuration file, trains a CNN model and visualizes its predictions on a held-out patient. For patient confidentiality reasons, we cannot include our CT data set in this repository. But we included a config.yml and ConfigReader class to facilitate reading in patient data.

In order to run the example script:
1. Insert your local data directories in config.yml.
2. (Optional) change network parameters in config.yml.
3. Call Deepy3d class:
```shell
python ./deepy3d/Deepy3d.py
```

### 3D-Slicer
The repository also contains a fork of a medical image processing and visualization package called [3D-Slicer](https://github.com/Slicer/Slicer). We used Slicer for manipulating CT-scans and performing global thresholding on the grayscale intensity values (Hounsfield units).

#### Installation
Install dependencies (Ubuntu):
```shell
sudo apt-get -y install unzip curl wget xz-utils bsdtar
```
Call Slicer installation script:
```shell
sh -c "./slicer/installSlicer.sh)"
```

#### Example script
Download sample data from slicer.org:
```shell
./tools/getSampleData.bash
```
Apply thresholding:
```shell
./tools/launchSlicer.sh examples/slicer_CT_threshold.py
```

## Contact
Bugs, questions and comments can be submitted in the issues tracker.
