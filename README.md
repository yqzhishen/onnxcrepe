# onnxcrepe

ONNX deployment of the CREPE [1] pitch tracker. The original Tensorflow
implementation can be found [here](https://github.com/marl/crepe/). The
provided model weights and most of the codes in this repository were exported
and converted from Max Morrison's [torchcrepe](https://github.com/maxrmorrison/torchcrepe),
a PyTorch implementation of CREPE.


## Usage
Documentation of this repository is still a work in progress and is
comming soon.


## Acknowledgements
Codes and model weights in this repository are based on the following repos:
- [torchcrepe](https://github.com/maxrmorrison/torchcrepe)
- [onnxruntime](https://github.com/microsoft/onnxruntime)
- [onnx-optimizer](https://github.com/onnx/optimizer)
- [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)


## References
[1] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, “Crepe: A
Convolutional Representation for Pitch Estimation,” in 2018 IEEE
International Conference on Acoustics, Speech and Signal
Processing (ICASSP).
