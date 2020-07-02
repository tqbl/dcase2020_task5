import oyaml as yaml
import pandas as pd

from core.dataset import Dataset
from core.dataset import DataSubset


class SONYC_UST(Dataset):
    def __init__(self, root_dir, annotations_path=None, version=2):
        super().__init__(f'SONYC-UST v{version}', root_dir, clip_duration=10)

        # Determine label set
        taxonomy_path = self.root_dir / 'dcase-ust-taxonomy.yaml'
        label_sets = _read_taxonomy(taxonomy_path)
        self.coarse_label_set, self.fine_label_set = label_sets
        self.label_set = self.fine_label_set

        # Read annotations (tags) from file
        if annotations_path is None or not annotations_path.name:
            annotations_path = self.root_dir / 'annotations.csv'
        dtype_map = {f'{label}_proximity': object
                     for label in self.fine_label_set}
        tags = pd.read_csv(annotations_path,
                           index_col='audio_filename',
                           dtype=dtype_map,
                           )

        # Create the relevant DataSubsets
        if version == 1:
            self._add_v1_subsets(tags)
        elif version == 2:
            self._add_v2_subsets(tags)
        else:
            raise ValueError(f'Unsupported dataset version: {version}')

    def _add_v1_subsets(self, tags):
        self.add_subset(DataSubset('training', self,
                                   tags[tags.split == 'train'],
                                   self.root_dir / 'train'))
        self.add_subset(DataSubset('validation', self,
                                   tags[tags.split == 'validate'],
                                   self.root_dir / 'validate'))
        self.add_subset(DataSubset('test', self,
                                   tags[tags.split == 'test'],
                                   self.root_dir / 'audio-eval'))

    def _add_v2_subsets(self, tags):
        audio_dir = self.root_dir / 'audio'
        subset = DataSubset('all', self, tags, audio_dir)
        self.add_subset(subset.subset('training', tags.split == 'train'))
        self.add_subset(subset.subset('validation', tags.split == 'validate'))
        self.add_subset(subset.subset('test', tags.split == 'test'))


def _read_taxonomy(path):
    with open(path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    coarse_labels = [f'{i}_{label}' for i, label in taxonomy['coarse'].items()]
    fine_labels = [f'{i}-{j}_{label}'
                   for i, group in taxonomy['fine'].items()
                   for j, label in group.items()]
    return coarse_labels, fine_labels


def labels_mean(subset):
    return _aggregate_labels(subset, pd.core.groupby.DataFrameGroupBy.mean)


def labels_max(subset):
    return _aggregate_labels(subset, pd.core.groupby.DataFrameGroupBy.max)


def _aggregate_labels(subset, operation):
    columns = [f'{label}_presence' for label in subset.dataset.label_set]
    return operation(subset.tags.filter(columns).groupby(level=0, sort=False))


class STCWrapper:
    def __init__(self, loader):
        self.loader = loader

    def __call__(self, subset):
        x, y = self.loader(subset)

        df = subset.tags.groupby(level=0, sort=False).first()
        features = [
            STCWrapper._one_hot(df['week'] // 4, n=14),
            STCWrapper._one_hot(df['hour'] // 3, n=8),
            (df['day'] > 4).astype(float),
            STCWrapper._triangle(df['week'], max_val=26),
            STCWrapper._triangle(df['hour'], max_val=12),
            df['day'],
        ]
        aux = pd.concat(features, axis=1).values

        return (x, aux), y

    def _one_hot(series, n):
        return pd.get_dummies(series).T.reindex(list(range(n))).T.fillna(0)

    def _triangle(x, max_val):
        return max_val - (x - max_val).abs()
