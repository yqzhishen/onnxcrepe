import numpy as np
import onnxcrepe


###############################################################################
# Test core.py
###############################################################################


def test_infer_tiny(frames, activation_tiny):
    """Test that inference is the same as the original crepe"""
    activation = onnxcrepe.infer(
        onnxcrepe.CrepeInferenceSession('tiny', providers=['CPUExecutionProvider']), frames)
    diff = np.abs(activation - activation_tiny)

    # ONNX output are not strictly the same as the original CREPE
    assert diff.max() < 0.5 and diff.mean() < 5e-3


def test_infer_full(frames, activation_full):
    """Test that inference is the same as the original crepe"""
    activation = onnxcrepe.infer(
        onnxcrepe.CrepeInferenceSession('full', providers=['CPUExecutionProvider']), frames)
    diff = np.abs(activation - activation_full)

    # ONNX output are not strictly the same as the original CREPE
    assert diff.max() < 0.5 and diff.mean() < 5e-3
