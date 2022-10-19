# onnxcrepe
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

ONNX deployment of the CREPE [1] pitch tracker. The provided model weights and most of the codes in this repository were converted and migrated from the original TensorFlow implementation [here](https://github.com/marl/crepe/) and Max Morrison's [torchcrepe](https://github.com/maxrmorrison/torchcrepe), a PyTorch implementation of CREPE.


## Usage

Download model weights from [releases](https://github.com/yqzhishen/onnxcrepe/releases) and put them into the `onnxcrepe/assets/` directory. See demo [here](samples/demo.py).

Documentation of this repository is still a work in progress and is comming soon.


## Acknowledgements
Codes and model weights in this repository are based on the following repos:
- [torchcrepe](https://github.com/maxrmorrison/torchcrepe) for 'full' and 'tiny' model weights and most of the code implementation
- [Weights_Keras_2_Pytorch](https://github.com/AgCl-LHY/Weights_Keras_2_Pytorch) for converting 'large', 'medium' and 'small' model weights from the original implementation
- [PyTorch](https://github.com/pytorch/pytorch) for exporting onnx models
- [onnx-optimizer](https://github.com/onnx/optimizer) and [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) for optimizing performance
- [onnxruntime](https://github.com/microsoft/onnxruntime) for execution and configurations


## References
[1] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, “[Crepe: A Convolutional Representation for Pitch Estimation](https://arxiv.org/abs/1802.06182),” in 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
