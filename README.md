# iPPG 2 BP: reconstructing blood pressure waves from imaging photoplethysmographic signals

This repository contains the source codes related to a deep learning model dedicated to the conversion of imaging PPG signals (computed from video) into BP signals (measured by a continuous non-invasive sensor). The technique has been evaluated on [BP4D+](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html).

## Reference
If you find this code useful or use it in an academic or research project, please cite it as:

Frédéric Bousefsaf et al., **iPPG 2 BP: reconstructing blood pressure waves from imaging photoplethysmographic signals using U-Net architectures**, *preprint*, 2021.

You can also visit my [website](https://sites.google.com/view/frederic-bousefsaf) for additional information.

## Scientific description
Please refer to the original publication to get all the details. We propose converting imaging photoplethysmographic (iPPG) to blood pressure (BP) signals using their continuous wavelet transforms (CWT). The real and imaginary parts of the CWT are passed to a deep pre-trained (ResNeXt101) U-shaped architecture.


![Alt text](illustrations/overview2.png?raw=true "Overview")


## Requirements
Deep learning models have been developed and learned through Tensorflow+Keras frameworks (2.4.0) over Python 3.5/3.6 . Results were analyzed with MATLAB R2020b.

Different packages must be installed to properly run the codes : 
- `pip install tensorflow` (or `tensorflow-gpu`)
- `pip install opencv-python`
- `pip install matplotlib`
- `pip install scipy`
- `pip install scikit-learn`
- `pip install segmentation-models`


## Usage
**Training**

<!---
`train and predict/train.py` includes all the procedures. The input, `data.mat`, corresponds to a collection of continuous wavelet representation (size: 256×256) of iPPG and ground truth BP signals (not supplied here). `train and predict/signal_to_cwt.py` is the MATLAB procedure dedicated to the conversion of a raw PPG signal (5 successive PPG waves) to its wavelet representation.
--->


**Prediction**

<!---
Trained architectures (U-Net supported by a ResNeXt101 backbone) [are freely available.](https://zenodo.org/record/5482374)

The dataset employed to train the networks has not been publicly released yet but excerpts [are available.](https://zenodo.org/record/5477689)

`train and predict/predict.py` will output a `.mat` file that can be analyzed with the `results analysis/main.py` MATLAB code.

![Alt text](illustrations/pred.png?raw=true "Results computed from sample data")
--->
