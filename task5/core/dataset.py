from pathlib import Path

import pandas as pd

from core.mask import FrameMask


class Dataset:
    def __init__(self, name, root_path, clip_duration=None):
        self.name = name
        self.root_path = Path(root_path)
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
        self.name = name
        self.dataset = dataset

        if tags is None:
            # Create an empty DataFrame if no tags are given
            index = pd.Index([path.name for path in audio_dir.iterdir()])
            self.tags = pd.DataFrame(index=index)
        else:
            self.tags = tags.copy()

        # Record the audio directory as a tag
        if '_audio_dir' not in self.tags.columns:
            self.tags['_audio_dir'] = audio_dir or dataset.root_path

    def concat(subsets, name=None):
        tags = pd.concat([subset.tags for subset in subsets])
        return DataSubset(name or subsets[0].name, subsets[0].dataset, tags)

    def audio_paths(self, must_exist=False):
        return (self.tags._audio_dir / self.tags.index).unique()

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
