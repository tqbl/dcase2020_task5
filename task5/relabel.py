import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import cli


def relabel(args):
    import pandas as pd
    import sklearn.metrics

    import core.sonyc_ust as sonyc_ust
    from core.dataloader import SimpleLabeler

    dataset = sonyc_ust.SONYC_UST(args.dataset_dir, version=2)
    train_set = dataset['training']
    val_set = dataset['validation']

    # Create ground truth using validation set and subset of training set
    labeler = SimpleLabeler(r'^\d-.*presence', regex=True)
    y_train1 = labeler(val_set['annotator_id==0'])
    y_train2 = labeler(train_set['annotator_id>=0'])
    count = args.minimum - y_train1.sum().astype(int)
    y_train2 = sample(y_train2, count)
    y_train = pd.concat([y_train1, y_train2])

    # Create corresponding input
    x_train1 = sonyc_ust.labels_mean(val_set['annotator_id>0'])
    x_train2 = sonyc_ust.labels_mean(train_set['annotator_id>0'])
    x_train = pd.concat([x_train1, x_train2]).loc[y_train.index]

    # Convert data into PyTorch tensors
    x_train = torch.FloatTensor(x_train.values)
    y_train = torch.FloatTensor(y_train.values)

    # Create validation set if applicable
    if args.validate:
        x_val = x_train[:100]
        y_val = y_train[:100]
        x_train = x_train[100:]
        y_train = y_train[100:]

    # Create and train model
    model = train(x_train, y_train, args.n_epochs)

    # Evaluate model on validation set
    if args.validate:
        acc_before = sklearn.metrics.accuracy_score(y_val, x_val.ceil())
        y_pred = model(x_val).sigmoid().detach().round()
        acc_after = (sklearn.metrics.accuracy_score(y_val, y_pred))
        print(f'Accuracy using max(): {acc_before}')
        print(f'Accuracy using pseudo-labels: {acc_after}')

    # Generate predictions and create DataFrame for pseudolabels
    # The validation set annotations are also included so that the
    # output file can replace the original annotations.csv file
    # during training (the test set is therefore not needed).
    x = sonyc_ust.labels_mean(train_set['annotator_id>=0'])
    y = model(torch.FloatTensor(x.values)).sigmoid().detach()
    df_train = train_set.tags.groupby(level=0).first()
    df_train.loc[:, x.columns] = y
    df = pd.concat([df_train, val_set.tags])
    del df['_audio_dir']

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path)


def sample(y, count):
    y_group = y.groupby(level=0)
    y_mean = y_group.mean()
    y_max = y_group.max()
    mask = y_max.index.isna()
    for k, column in enumerate(y_max):
        if count[k] > 0:
            y_k = y_mean[column]
            y_k = y_k.sort_values(ascending=False).iloc[:count[k]]
            mask |= y_max.index.isin(y_k.index)
    return y_max[mask]


def train(x_train, y_train, n_epochs=25):
    torch.manual_seed(1000)

    # Instantiate neural network
    model = MLP(y_train.shape[-1])

    # Use cross-entropy loss function and Adam optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.02)

    # Use helper class to iterate over data in batches
    loader = DataLoader(TensorDataset(x_train, y_train),
                        batch_size=32, shuffle=True)

    for epoch in range(n_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    return model.eval()


class MLP(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(in_features=n_classes, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())


def parse_args():
    config, conf_parser, remaining_args = cli.parse_config_args()
    parser = argparse.ArgumentParser(parents=[conf_parser])

    args_default = dict(config.items('Default'))
    output_path = config['Training']['pseudolabel_path']
    parser.set_defaults(**args_default, output_path=output_path)
    parser.add_argument('--dataset_dir', type=Path, metavar='DIR')
    parser.add_argument('--minimum', type=int, default=30, metavar='N')
    parser.add_argument('--n_epochs', type=int, default=20, metavar='N')
    parser.add_argument('--validate', type=cli.boolean, metavar='BOOL')
    parser.add_argument('--output_path', type=Path, metavar='PATH')

    return parser.parse_args(remaining_args)


if __name__ == '__main__':
    sys.exit(relabel(parse_args()))
