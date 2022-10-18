import numpy as np

import onnxcrepe
from onnxcrepe.session import CrepeInferenceSession

# Load audio
audio, sr = onnxcrepe.load.audio(r'assets/xtgg_mono_16k_denoise.wav')

# Here we'll use a 5 millisecond hop length
hop_length = int(sr / 200.)

# Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
# This would be a reasonable range for speech
fmin = 50
fmax = 1100

# Select a model capacity--one of "tiny" or "full"
model = 'full'

# Choose execution providers to use for inference
providers = ['DmlExecutionProvider', 'CPUExecutionProvider']

# Pick a batch size that doesn't cause memory errors on your gpu
batch_size = 512

# Create inference session
session = CrepeInferenceSession(
    model='full',
    providers=providers)

# Compute pitch using the default DirectML GPU or CPU
pitch = onnxcrepe.predict(session, audio, sr, hop_length=hop_length, fmin=fmin, fmax=fmax, batch_size=batch_size)
print(pitch.shape)
print(np.mean(pitch))
print(np.var(pitch))
