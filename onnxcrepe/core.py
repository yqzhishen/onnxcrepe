import librosa
import numpy as np
import tqdm

import onnxcrepe

__all__ = ['CENTS_PER_BIN',
           'MAX_FMAX',
           'PITCH_BINS',
           'SAMPLE_RATE',
           'WINDOW_SIZE',
           'UNVOICED',
           'predict',
           'predict_from_file',
           'predict_from_file_to_file',
           'predict_from_files_to_files',
           'preprocess',
           'infer',
           'postprocess',
           'resample']

###############################################################################
# Constants
###############################################################################


CENTS_PER_BIN = 20  # cents
MAX_FMAX = 2006.  # hz
PITCH_BINS = 360
SAMPLE_RATE = 16000  # hz
WINDOW_SIZE = 1024  # samples
UNVOICED = np.nan


###############################################################################
# Crepe pitch prediction
###############################################################################


def predict(session,
            audio,
            sample_rate,
            hop_length=None,
            fmin=50.,
            fmax=MAX_FMAX,
            decoder=onnxcrepe.decode.viterbi,
            return_periodicity=False,
            batch_size=None,
            pad=True):
    """Performs pitch estimation

    Arguments
        session (onnxcrepe.CrepeInferenceSession)
            An onnxruntime.InferenceSession holding the CREPE model
        audio (numpy.ndarray [shape=(n_samples,)])
            The audio signal
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        decoder (function)
            The decoder to use. See decode.py for decoders.
        return_periodicity (bool)
            Whether to also return the network confidence
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        pitch (numpy.ndarray [shape=(1, 1 + int(time // hop_length))])
        (Optional) periodicity (numpy.ndarray
                                [shape=(1, 1 + int(time // hop_length))])
    """

    results = []

    # Preprocess audio
    generator = preprocess(audio,
                           sample_rate,
                           hop_length,
                           batch_size,
                           pad)
    for frames in generator:

        # Infer independent probabilities for each pitch bin
        probabilities = infer(session, frames)  # shape=(batch, 360)

        probabilities = probabilities.transpose(1, 0)[None]  # shape=(1, 360, batch)

        # Convert probabilities to F0 and periodicity
        result = postprocess(probabilities,
                             fmin,
                             fmax,
                             decoder,
                             return_periodicity)

        # Place on same device as audio to allow very long inputs
        if isinstance(result, tuple):
            result = (result[0], result[1])

        results.append(result)

    # Split pitch and periodicity
    if return_periodicity:
        pitch, periodicity = zip(*results)
        return np.concatenate(pitch, axis=1), np.concatenate(periodicity, axis=1)

    # Concatenate
    return np.concatenate(results, axis=1)


def predict_from_file(session,
                      audio_file,
                      hop_length=None,
                      fmin=50.,
                      fmax=MAX_FMAX,
                      decoder=onnxcrepe.decode.viterbi,
                      return_periodicity=False,
                      batch_size=None,
                      pad=True):
    """Performs pitch estimation from file on disk

    Arguments
        session (onnxcrepe.CrepeInferenceSession)
            An onnxruntime.InferenceSession holding the CREPE model
        audio_file (string)
            The file to perform pitch tracking on
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        decoder (function)
            The decoder to use. See decode.py for decoders.
        return_harmonicity (bool) [DEPRECATED]
            Whether to also return the network confidence
        return_periodicity (bool)
            Whether to also return the network confidence
        batch_size (int)
            The number of frames per batch
        device (string)
            The device used to run inference
        pad (bool)
            Whether to zero-pad the audio

    Returns
        pitch (numpy.ndarray [shape=(1, 1 + int(time // hop_length))])
        (Optional) periodicity (torch.tensor
                                [shape=(1, 1 + int(time // hop_length))])
    """
    # Load audio
    audio, sample_rate = onnxcrepe.load.audio(audio_file)

    # Predict
    return predict(session, audio, sample_rate, hop_length, fmin, fmax, decoder, return_periodicity, batch_size, pad)


def predict_from_file_to_file(session,
                              audio_file,
                              output_pitch_file,
                              output_periodicity_file=None,
                              hop_length=None,
                              fmin=50.,
                              fmax=MAX_FMAX,
                              decoder=onnxcrepe.decode.viterbi,
                              batch_size=None,
                              pad=True):
    """Performs pitch estimation from file on disk

    Arguments
        session (onnxcrepe.CrepeInferenceSession)
            An onnxruntime.InferenceSession holding the CREPE model
        audio_file (string)
            The file to perform pitch tracking on
        output_pitch_file (string)
            The file to save predicted pitch
        output_periodicity_file (string or None)
            The file to save predicted periodicity
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        decoder (function)
            The decoder to use. See decode.py for decoders.
        batch_size (int)
            The number of frames per batch
        device (string)
            The device used to run inference
        pad (bool)
            Whether to zero-pad the audio
    """

    # Predict from file
    prediction = predict_from_file(hop_length, audio_file, session, fmin, fmax, decoder,
                                   output_periodicity_file is not None, batch_size, pad)

    # Save to disk
    if output_periodicity_file is not None:
        np.save(output_pitch_file, prediction[0])
        np.save(output_periodicity_file, prediction[1])
    else:
        np.save(output_pitch_file, prediction)


def predict_from_files_to_files(session,
                                audio_files,
                                output_pitch_files,
                                output_periodicity_files=None,
                                hop_length=None,
                                fmin=50.,
                                fmax=MAX_FMAX,
                                decoder=onnxcrepe.decode.viterbi,
                                batch_size=None,
                                pad=True):
    """Performs pitch estimation from files on disk without reloading model

    Arguments
        session (onnxcrepe.CrepeInferenceSession)
            An onnxruntime.InferenceSession holding the CREPE model
        audio_files (list[string])
            The files to perform pitch tracking on
        output_pitch_files (list[string])
            The files to save predicted pitch
        output_periodicity_files (list[string] or None)
            The files to save predicted periodicity
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        decoder (function)
            The decoder to use. See decode.py for decoders.
        batch_size (int)
            The number of frames per batch
        device (string)
            The device used to run inference
        pad (bool)
            Whether to zero-pad the audio
    """

    if output_periodicity_files is None:
        output_periodicity_files = len(audio_files) * [None]

    # Setup iterator
    iterator = zip(audio_files, output_pitch_files, output_periodicity_files)
    iterator = tqdm.tqdm(iterator, desc='onnxcrepe', dynamic_ncols=True)
    for audio_file, output_pitch_file, output_periodicity_file in iterator:
        # Predict a file
        predict_from_file_to_file(session, audio_file, output_pitch_file, output_periodicity_file, hop_length, fmin,
                                  fmax, decoder, batch_size, pad)


###############################################################################
# Components for step-by-step prediction
###############################################################################


def preprocess(audio,
               sample_rate,
               hop_length=None,
               batch_size=None,
               pad=True):
    """Convert audio to model input

    Arguments
        audio (numpy.ndarray [shape=(time,)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        frames (numpy.ndarray [shape=(1 + int(time // hop_length), 1024)])
    """
    # Default hop length of 10 ms
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    # Resample
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, sample_rate, SAMPLE_RATE)
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)

    # Get total number of frames

    # Maybe pad
    if pad:
        total_frames = 1 + int(audio.shape[0] // hop_length)
        audio = np.pad(
            audio,
            (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    else:
        total_frames = 1 + int((audio.shape[0] - WINDOW_SIZE) // hop_length)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):
        # Batch indices
        start = max(0, i * hop_length)
        end = min(audio.shape[0],
                  (i + batch_size - 1) * hop_length + WINDOW_SIZE)

        # Chunk
        n_bytes = audio.strides[-1]
        frames = np.lib.stride_tricks.as_strided(
            audio[start:end],
            shape=((end - start - WINDOW_SIZE) // hop_length, WINDOW_SIZE),
            strides=(hop_length * n_bytes, n_bytes))  # shape=(batch, 1024)

        # Note:
        # Z-score standardization operations originally located here
        # (https://github.com/maxrmorrison/torchcrepe/blob/master/torchcrepe/core.py#L692)
        # are wrapped into the ONNX models for hardware acceleration.

        yield frames


def infer(session, frames):
    """Forward pass through the model

    Arguments
        session (onnxcrepe.CrepeInferenceSession)
            An onnxruntime.InferenceSession holding the CREPE model
        frames (numpy.ndarray [shape=(time / hop_length, 1024)])
            The network input

    Returns
        logits (numpy.ndarray [shape=(1 + int(time // hop_length), 360)])
    """
    # Apply model
    return session.run(None, {'frames': frames})[0]


def postprocess(probabilities,
                fmin=0.,
                fmax=MAX_FMAX,
                decoder=onnxcrepe.decode.viterbi,
                return_periodicity=False):
    """Convert model output to F0 and periodicity

    Arguments
        probabilities (numpy.ndarray [shape=(1, 360, time / hop_length)])
            The probabilities for each pitch bin inferred by the network
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        viterbi (bool)
            Whether to use viterbi decoding
        return_harmonicity (bool) [DEPRECATED]
            Whether to also return the network confidence
        return_periodicity (bool)
            Whether to also return the network confidence

    Returns
        pitch (numpy.ndarray [shape=(1, 1 + int(time // hop_length))])
        periodicity (numpy.ndarray [shape=(1, 1 + int(time // hop_length))])
    """
    # Convert frequency range to pitch bin range
    minidx = onnxcrepe.convert.frequency_to_bins(fmin)
    maxidx = onnxcrepe.convert.frequency_to_bins(fmax, np.ceil)

    # Remove frequencies outside allowable range
    probabilities[:, :minidx] = float('-inf')
    probabilities[:, maxidx:] = float('-inf')

    # Perform argmax or viterbi sampling
    bins, pitch = decoder(probabilities)

    if not return_periodicity:
        return pitch

    # Compute periodicity from probabilities and decoded pitch bins
    return pitch, periodicity(probabilities, bins)


###############################################################################
# Utilities
###############################################################################


def periodicity(probabilities, bins):
    """Computes the periodicity from the network output and pitch bins"""
    # shape=(batch * time / hop_length, 360)
    probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)

    # shape=(batch * time / hop_length, 1)
    bins_stacked = bins.reshape(-1, 1).astype(np.int64)

    # Use maximum logit over pitch bins as periodicity
    periodicity = np.take_along_axis(probs_stacked, bins_stacked, axis=1)

    # shape=(batch, time / hop_length)
    return periodicity.reshape(probabilities.shape[0], probabilities.shape[2])


def resample(audio, sample_rate):
    """Resample audio"""
    return librosa.resample(audio, sample_rate, onnxcrepe.SAMPLE_RATE)
