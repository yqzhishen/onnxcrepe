import os.path

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
            precision=None,
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
        precision (float)
            The precision in milliseconds, i.e. the length of each frame
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
        pitch (numpy.ndarray [shape=(1, 1 + int(time // precision))])
        (Optional) periodicity (numpy.ndarray
                                [shape=(1, 1 + int(time // precision))])
    """

    results = []

    # Preprocess audio
    generator = preprocess(audio,
                           sample_rate,
                           precision,
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
                      precision=None,
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
        precision (float)
            The precision in milliseconds, i.e. the length of each frame
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
        pitch (numpy.ndarray [shape=(1, 1 + int(time // precision))])
        (Optional) periodicity (torch.tensor
                                [shape=(1, 1 + int(time // precision))])
    """
    # Load audio
    audio, sample_rate = onnxcrepe.load.audio(audio_file)

    # Predict
    return predict(session, audio, sample_rate, precision, fmin, fmax, decoder, return_periodicity, batch_size, pad)


def predict_from_file_to_file(session,
                              audio_file,
                              output_directory=None,
                              save_periodicity=False,
                              format='csv',
                              precision=None,
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
        output_directory (string or None)
            The directory to save results.
            None means saving results in the same directory as the audio file.
        save_periodicity (bool)
            Whether save predicted periodicity
        format (string)
            The output format. 'csv' means combined csv file and
            'npy' means separated npy files (pitch and periodicity).
        precision (float)
            The precision in milliseconds, i.e. the length of each frame
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
    prediction = predict_from_file(session, audio_file, precision, fmin, fmax, decoder,
                                   save_periodicity, batch_size, pad)

    # Get audio filename without extension
    title = os.path.basename(audio_file).rsplit('.', maxsplit=1)[0]

    # Get output directory
    if output_directory is None:
        output_directory = os.path.dirname(audio_file)

    # Save to disk
    if format == 'csv':
        with open(os.path.join(output_directory, f'{title}.pitch.csv'), 'w') as f:
            if save_periodicity:
                for i in range(prediction[0].shape[1]):
                    # time, f0, periodicity
                    print('%f,%f,%f'
                          % (i * precision / 1000., prediction[0][0][i], prediction[1][0][i]),
                          file=f)
            else:
                for i in range(prediction.shape[1]):
                    # time, f0
                    print('%f,%f'
                          % (i * precision / 1000., prediction[0][i]),
                          file=f)
    elif format == 'npy':
        np.save(os.path.join(output_directory, f'{title}.f0.npy'), prediction[0])
        if save_periodicity:
            np.save(os.path.join(output_directory, f'{title}.periodicity.npy'), prediction[1])


def predict_from_files_to_files(session,
                                audio_files,
                                output_directory=None,
                                save_periodicity=False,
                                format='csv',
                                precision=None,
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
        output_directory (string or None)
            The directory to save results.
            None means saving results in the same directory as each audio file.
        save_periodicity (bool)
            Whether save predicted periodicity
        format (string)
            The output format. 'csv' means combined csv file and
            'npy' means separated npy files (pitch and periodicity).
        precision (float)
            The precision in milliseconds, i.e. the length of each frame
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

    # Setup iterator
    iterator = tqdm.tqdm(audio_files, desc='onnxcrepe', dynamic_ncols=True)
    for audio_file in iterator:
        # Predict a file
        predict_from_file_to_file(session, audio_file, output_directory, save_periodicity, format, precision, fmin,
                                  fmax, decoder, batch_size, pad)


###############################################################################
# Components for step-by-step prediction
###############################################################################


def preprocess(audio,
               sample_rate,
               precision=None,
               batch_size=None,
               pad=True):
    """Convert audio to model input

    Arguments
        audio (numpy.ndarray [shape=(time,)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        precision (float)
            The precision in milliseconds, i.e. the length of each frame
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        frames (numpy.ndarray [shape=(1 + int(time // precision), 1024)])
    """
    # Resample
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    # Default hop length of 10 ms
    hop_length = SAMPLE_RATE / 100 if precision is None else SAMPLE_RATE * precision / 1000

    # Get total number of frames

    # Maybe pad
    if pad:
        total_frames = 1 + int(audio.shape[0] / hop_length)
        audio = np.pad(
            audio,
            (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    else:
        total_frames = 1 + int((audio.shape[0] - WINDOW_SIZE) / hop_length)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):
        # Batch indices
        start = max(0, int(i * hop_length))
        end = min(audio.shape[0],
                  int((i + batch_size - 1) * hop_length) + WINDOW_SIZE)

        # Chunk
        n_bytes = audio.strides[-1]
        frames = np.lib.stride_tricks.as_strided(
            audio[start:end],
            shape=((end - start - WINDOW_SIZE) // int(hop_length) + 1, WINDOW_SIZE),
            strides=(int(hop_length) * n_bytes, n_bytes))  # shape=(batch, 1024)

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
        frames (numpy.ndarray [shape=(time / precision, 1024)])
            The network input

    Returns
        logits (numpy.ndarray [shape=(1 + int(time // precision), 360)])
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
        probabilities (numpy.ndarray [shape=(1, 360, time / precision)])
            The probabilities for each pitch bin inferred by the network
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        decoder (function)
            The decoder to use. See decode.py for decoders.
        return_periodicity (bool)
            Whether to also return the network confidence

    Returns
        pitch (numpy.ndarray [shape=(1, 1 + int(time // precision))])
        periodicity (numpy.ndarray [shape=(1, 1 + int(time // precision))])
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
    # shape=(time / precision, 360)
    probs_stacked = probabilities.transpose(0, 2, 1).reshape(-1, PITCH_BINS)
    # shape=(time / precision, 1)
    bins_stacked = bins.reshape(-1, 1).astype(np.int64)

    # Use maximum logit over pitch bins as periodicity
    periodicity = np.take_along_axis(probs_stacked, bins_stacked, axis=1)

    # shape=(batch, time / precision)
    return periodicity.reshape(probabilities.shape[0], probabilities.shape[2])


def resample(audio, sample_rate):
    """Resample audio"""
    return librosa.resample(audio, orig_sr=sample_rate, target_sr=onnxcrepe.SAMPLE_RATE)
