from pathlib import Path

import pandas as pd

from core.mask import FrameMask


class Dataset:
    def __init__(self, name, root_dir, clip_duration=None):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f'No such directory: {root_dir}')

        self.name = name
        self.clip_duration = clip_duration
        self.subsets = dict()

    def add_subset(self, subset):
        self.subsets[subset.name] = subset

    def __getitem__(self, key):
        return self.subsets[key]

    def __setitem__(self, key, value):
        self.subsets[key] = value

    def __delitem__(self, key):
        del self.subsets[key]

    def __iter__(self):
        return iter(self.subsets)

    def __len__(self):
        return len(self.subsets)

    def __str__(self):
        return self.name


class DataSubset:
    def __init__(self, name, dataset, tags=None, audio_dir=None):
        if audio_dir is None:
            audio_dir = dataset.root_dir
        else:
            audio_dir = Path(audio_dir)
            if not audio_dir.is_dir():
                raise FileNotFoundError(f'No such directory: {audio_dir}')

        self.name = name
        self.dataset = dataset
        self._audio_paths = None

        if tags is None:
            # Create an empty DataFrame if no tags are given
            index = pd.Index([path.name for path in audio_dir.iterdir()])
            self.tags = pd.DataFrame(index=index)
        else:
            self.tags = tags.copy()

        # Record the audio directory as a tag
        if '_audio_dir' not in self.tags.columns:
            self.tags['_audio_dir'] = audio_dir

    def concat(subsets, name=None):
        tags = pd.concat([subset.tags for subset in subsets])
        return DataSubset(name or subsets[0].name, subsets[0].dataset, tags)

    @property
    def audio_paths(self):
        if self._audio_paths is None:
            audio_dirs = self.tags._audio_dir.groupby(
                level=0, sort=False).first()
            fnames = self.tags.index.unique(level=0)
            self._audio_paths = audio_dirs / fnames

        return self._audio_paths

    def subset(self, name, mask, complement=False):
        if callable(mask):
            mask = mask(self.tags)
        elif isinstance(mask, str):
            mask = FrameMask(mask).value(self.tags)
        elif isinstance(mask, FrameMask):
            mask = mask.value(self.tags)

        if complement:
            mask = ~mask

        return DataSubset(name, self.dataset, self.tags[mask])

    def subset_loc(self, name, index):
        return DataSubset(name, self.dataset, self.tags.loc[index])

    def subset_iloc(self, name, indexer):
        return DataSubset(name, self.dataset, self.tags.iloc[indexer])

    def __getitem__(self, key):
        return self.subset(self.name, key)

    def __len__(self):
        return len(self.tags)

    def __str__(self):
        return f'{self.dataset} {self.name}'
