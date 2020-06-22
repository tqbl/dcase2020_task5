import argparse
import sys
from pathlib import Path

import cli
import utils


def train(args):
    import core.sonyc_ust as sonyc_ust
    import pytorch.training as training
    from core.dataloader import SpectrogramLoader, Standardizer, STCWrapper

    dataset = sonyc_ust.SONYC_UST(args.dataset_dir, version=2)
    train_set = dataset['training']
    val_set = dataset['validation']

    # Mask out data based on user specification
    if args.training_mask:
        train_set = train_set[args.training_mask]
    if args.validation_mask:
        val_set = val_set[args.validation_mask]

    # Load training and validation data
    labeler = sonyc_ust.labels_max
    loader = Standardizer(
        SpectrogramLoader(args.extraction_dir, labeler),
        stats_path=args.extraction_dir / 'stats.p',
        axis=(0, 1, 2),
    )
    if args.use_stc:
        loader = STCWrapper(loader)  # Load STC metadata too
    x_train, y_train = utils.load_data(train_set, loader)
    x_val, y_val = utils.load_data(val_set, loader)

    # Ensure output directories exist
    log_dir = args.log_dir / args.training_id
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir / args.training_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save hyperparameters to disk
    params = {
        'model': args.model,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'lr_decay': args.lr_decay,
        'lr_decay_rate': args.lr_decay_rate,
        'use_stc': args.use_stc,
        'augment': args.augment,
        'overwrite': args.overwrite,
    }
    masks = {
        'training_mask': str(args.training_mask),
        'validation_mask': str(args.validation_mask),
    }
    utils.log_parameters(args.log_dir / 'parameters.json', **params, **masks)

    training.train(x_train, y_train, x_val, y_val,
                   log_dir, model_dir, **params)


def _load_data(subset, data_loader):
    n = len(subset.tags.index.unique())
    print(f'Loading {subset.name} set ({n} instances)...')
    return utils.timeit(lambda: data_loader(subset),
                        f'Loaded {subset.name} set')


def parse_args():
    config, conf_parser, remaining_args = cli.parse_config_args()
    parser = argparse.ArgumentParser(parents=[conf_parser])

    args_default = dict(config.items('Default'))
    args_training = dict(config.items('Training'))
    parser.set_defaults(**args_default, **args_training)

    parser.add_argument('--dataset_dir', type=Path, metavar='DIR')
    parser.add_argument('--extraction_dir', type=Path, metavar='DIR')
    parser.add_argument('--model_dir', type=Path, metavar='DIR')
    parser.add_argument('--log_dir', type=Path, metavar='DIR')
    parser.add_argument('--training_id', metavar='ID')
    parser.add_argument('--model', choices=['gcnn', 'qkcnn10'])
    parser.add_argument('--training_mask', type=cli.mask, metavar='MASK')
    parser.add_argument('--validation_mask', type=cli.mask, metavar='MASK')
    parser.add_argument('--seed', type=int, metavar='N')
    parser.add_argument('--batch_size', type=int, metavar='N')
    parser.add_argument('--n_epochs', type=int, metavar='N')
    parser.add_argument('--lr', type=float, metavar='NUM')
    parser.add_argument('--lr_decay', type=float, metavar='NUM')
    parser.add_argument('--lr_decay_rate', type=int, metavar='N')
    parser.add_argument('--use_stc', type=cli.boolean, metavar='BOOL')
    parser.add_argument('--augment', type=cli.boolean, metavar='BOOL')
    parser.add_argument('--overwrite', type=cli.boolean, metavar='BOOL')

    return parser.parse_args(remaining_args)


if __name__ == '__main__':
    sys.exit(train(parse_args()))
