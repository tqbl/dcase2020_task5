import ast
from datetime import datetime

import h5py
import librosa
import numpy as np


def extract(audio_paths, extractor, output_path,
            clip_duration=None, overwrite=False):
    mode = 'w' if overwrite else 'a'
    with h5py.File(output_path, mode) as f:
        # Create/load the relevant HDF5 datasets
        size = len(audio_paths)
        str_dtype = h5py.string_dtype(encoding='utf-8')
        timestamps = f.require_dataset('timestamps', (size,), str_dtype)
        if clip_duration is None:
            dtype = h5py.vlen_dtype('float32')
            feats = f.require_dataset('F', (size,), dtype)

            # Record shape of reference feature vector. Used to infer
            # the original shape of a vector prior to flattening.
            feats.attrs['shape'] = extractor.output_shape(1)[1:]
        else:
            shape = (size,) + extractor.output_shape(clip_duration)
            feats = f.require_dataset('F', shape, dtype='float32')

        indexes = dict()
        for i, path in enumerate(audio_paths):
            # Associate index of feature vector with file name
            indexes[path.name] = i

            # Skip if feature vector exists and should not be recomputed
            if timestamps[i] and not overwrite:
                continue

            # Extract feature vector
            x, sample_rate = librosa.load(
                path, sr=None, duration=clip_duration)
            if clip_duration is None:
                feats[i] = extractor.extract(x, sample_rate).flatten()
            else:
                x = librosa.util.fix_length(x, sample_rate * clip_duration)
                feats[i] = extractor.extract(x, sample_rate)

            # Record timestamp in ISO format
            timestamps[i] = datetime.now().isoformat()

        # Store `indexes` dictionary as a string
        f.require_dataset('indexes', (), dtype=str_dtype)
        if indexes:
            indexes_prev = ast.literal_eval(f['indexes'][()] or '{}')
            f['indexes'][()] = str({**indexes_prev, **indexes})


def load(path, file_names=None):
    with h5py.File(path, 'r') as f:
        # Determine the corresponding indexes for each file name
        mapping = ast.literal_eval(f['indexes'][()])
        indexes = np.array([mapping[name] for name in file_names])

        # Ensure indexes are in ascending order for h5py indexing
        # Reverse the permutation after loading the h5py dataset subset
        sort_indexes = indexes.argsort()
        unsort_indexes = sort_indexes.argsort()
        indexes = indexes[sort_indexes]
        x = np.array(f['F'][indexes])[unsort_indexes]

        shape = f['F'].attrs.get('shape')
        return np.array(x), shape


class LogmelExtractor:
    def __init__(self,
                 sample_rate=32000,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=64,
                 ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Create Mel filterbank matrix
        self.mel_fb = librosa.filters.mel(sr=sample_rate,
                                          n_fft=n_fft,
                                          n_mels=n_mels,
                                          )

    def output_shape(self, clip_duration):
        n_samples = clip_duration * self.sample_rate
        n_frames = n_samples // self.hop_length + 1
        return (n_frames, self.mel_fb.shape[0])

    def extract(self, x, sample_rate):
        # Resample to target sample rate
        x = librosa.resample(x, sample_rate, self.sample_rate)

        # Compute mel-scaled spectrogram
        D = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.dot(self.mel_fb, np.abs(D) ** 2).T
        # Apply log non-linearity
        return librosa.power_to_db(S, ref=0., top_db=None)
