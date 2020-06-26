from pathlib import Path

import numpy as np
import pandas as pd
import pickle

import core.features as features


class SpectrogramLoader:
    def __init__(self, data_dir, labeler=None):
        self.data_dir = Path(data_dir)
        self.labeler = labeler or _empty_frame

    def __call__(self, subset):
        x, _ = _load_hdf5(self.data_dir, subset)
        x = np.expand_dims(x, axis=-1)  # NxTxF -> NxTxFx1
        y = self.labeler(subset)
        return x, y


class Standardizer:
    def __init__(self, loader, stats_path, axis):
        self.loader = loader
        self.stats_path = Path(stats_path)
        self.axis = axis

    def __call__(self, subset):
        x, y = self.loader(subset)

        if self.stats_path.exists():
            mean, std = pickle.load(open(self.stats_path, 'rb'))
        else:
            mean = x.mean(self.axis)
            std = x.std(self.axis)
            pickle.dump((mean, std), open(self.stats_path, 'wb'))
        x = (x - mean) / std

        return x, y


class SimpleLabeler:
    def __init__(self, pattern=None, regex=False, get_dummies=False):
        if pattern:
            self.pattern = pattern
            if regex:
                self.callback = self._labels_from_filtered
            else:
                self.callback = self._labels_from_column
        else:
            self.callback = self._labels

        self.get_dummies = get_dummies

    def __call__(self, subset):
        if self.get_dummies:
            return self._dummies(subset)
        return self.callback(subset)

    def _labels(self, subset):
        return subset.tags

    def _labels_from_column(self, subset):
        return subset.tags[self.pattern]

    def _labels_from_filtered(self, subset):
        return subset.tags.filter(regex=self.pattern)

    def _dummies(self, subset):
        dtype = pd.CategoricalDtype(categories=subset.dataset.label_set)
        return pd.get_dummies(self.callback(subset).astype(dtype))


def _empty_frame(subset):
    return pd.DataFrame(index=subset.tags.index.unique())


def _load_hdf5(data_dir, subset):
    data_path = data_dir / (subset.name + '.h5')
    index = subset.tags.index.unique()
    return features.load(data_path, index)
