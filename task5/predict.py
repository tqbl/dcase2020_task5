import argparse
import sys
from pathlib import Path

import cli
import utils


def predict(args):
    import pandas as pd

    import pytorch.training as training
    from core.dataloader import SpectrogramLoader, Standardizer
    from core.sonyc_ust import SONYC_UST, STCWrapper

    dataset = SONYC_UST(args.dataset_dir, version=2)
    subset = dataset[args.dataset]

    # Mask out data based on user specification
    if args.mask:
        subset = subset[args.mask]

    # Load input data
    loader = Standardizer(
        SpectrogramLoader(args.extraction_dir),
        stats_path=args.extraction_dir / 'stats.p',
        axis=(0, 1, 2),
    )
    if args.use_stc:
        loader = STCWrapper(loader)  # Load STC metadata too
    x, fnames = utils.load_data(subset, loader)

    # Compute predictions for each model and ensemble using mean
    log_dir = args.log_dir / args.training_id
    model_dir = args.model_dir / args.training_id
    epochs = _determine_epochs(args.epochs, log_dir)
    def _predict(epoch):
        y = training.predict(x, epoch, model_dir, args.use_stc)
        return pd.DataFrame(y, fnames.index, subset.dataset.label_set)
    preds = [utils.timeit(lambda: _predict(epoch),
                          f'[Epoch {epoch}] Computed predictions')
             for epoch in epochs]

    y_pred = pd.concat(preds).groupby(level=0).mean()

    # Compute and append coarse-level predictions
    y_pred.columns = subset.dataset.label_set
    y_pred_coarse = y_pred.T.groupby(y_pred.columns.str[0]).max().T
    y_pred_coarse.columns = subset.dataset.coarse_label_set
    y_pred = y_pred.join(y_pred_coarse)

    # Ensure output directory exists
    prediction_dir = args.prediction_dir / args.training_id
    prediction_dir.mkdir(parents=True, exist_ok=True)

    # Write predictions and hyperparameters to output directory
    output_path = prediction_dir / f'{subset.name}.csv'
    print(f'Output path: {output_path}')
    y_pred.to_csv(output_path)
    utils.log_parameters(prediction_dir / 'parameters.json',
                         epochs=args.epochs,
                         )

    # Remove model files that were not used for prediction
    if args.clean:
        count = 0
        for path in model_dir.glob('model.[0-9][0-9].pth'):
            if int(str(path)[-6:-4]) not in epochs:
                path.unlink()
                count += 1
        print(f'Removed {count} unused model files')


def parse_args():
    config, conf_parser, remaining_args = cli.parse_config_args()
    parser = argparse.ArgumentParser(parents=[conf_parser])

    args_default = dict(config.items('Default'))
    args_prediction = dict(config.items('Prediction'))
    training_id = config['Training']['training_id']
    use_stc = config['Training']['use_stc']
    parser.set_defaults(**args_default, **args_prediction,
                        training_id=training_id, use_stc=use_stc)

    parser.add_argument('dataset', choices=['training', 'validation', 'test'])
    parser.add_argument('--dataset_dir', type=Path, metavar='DIR')
    parser.add_argument('--extraction_dir', type=Path, metavar='DIR')
    parser.add_argument('--model_dir', type=Path, metavar='DIR')
    parser.add_argument('--log_dir', type=Path, metavar='DIR')
    parser.add_argument('--prediction_dir', type=Path, metavar='DIR')
    parser.add_argument('--training_id', metavar='ID')
    parser.add_argument('--use_stc', type=cli.boolean, metavar='BOOL')
    parser.add_argument('--mask', type=cli.mask, metavar='MASK')
    parser.add_argument('--epochs', type=_epochs, metavar='EPOCHS')
    parser.add_argument('--clean', type=cli.boolean, metavar='BOOL')

    return parser.parse_args(remaining_args)


def _epochs(arg):
    split = arg.split(':')

    if len(split) == 1:
        return list(map(int, arg.split(',')))

    metric, n_epochs = split
    if metric in ['val_loss',
                  'val_auprc_micro',
                  'val_auprc_macro',
                  'val_f1_micro',
                  ]:
        return metric, int(n_epochs)
    raise argparse.ArgumentTypeError(f'unrecognized metric: {metric}')


def _determine_epochs(spec, log_dir):
    import pandas as pd

    if type(spec) is list:
        return spec

    metric, n_epochs = spec
    df = pd.read_csv(log_dir / 'history.csv', index_col=0)
    df.sort_values(by=metric, ascending=metric in ['val_loss'], inplace=True)
    return df.index.values[:n_epochs]


if __name__ == '__main__':
    sys.exit(predict(parse_args()))
