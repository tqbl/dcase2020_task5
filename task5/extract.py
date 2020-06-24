import argparse
import sys
from pathlib import Path

import cli
import utils


def extract(args):
    from tqdm import tqdm

    import core.features as features
    from core.sonyc_ust import SONYC_UST

    dataset = SONYC_UST(args.dataset_dir, version=2)
    subset = dataset[args.dataset]

    # Use a logmel representation for feature extraction
    params = {
        'sample_rate': args.sample_rate,
        'n_fft': args.n_fft,
        'hop_length': args.hop_length,
        'n_mels': args.n_mels,
    }
    extractor = features.LogmelExtractor(**params)

    # Ensure output directory exists
    args.extraction_dir.mkdir(parents=True, exist_ok=True)

    # Write hyperparameters to disk
    utils.log_parameters(args.extraction_dir / 'parameters.json', **params)

    output_path = args.extraction_dir / (subset.name + '.h5')
    clip_duration = subset.dataset.clip_duration
    shape = extractor.output_shape(clip_duration or 0)
    if clip_duration is None:
        shape = (-1,) + shape[1:]
    print(f'Extracting features with shape {shape}...')
    print(f'Output path: {output_path}')
    audio_paths = tqdm(subset.audio_paths)
    features.extract(audio_paths,
                     extractor,
                     output_path,
                     clip_duration,
                     args.overwrite,
                     )


def parse_args():
    config, conf_parser, remaining_args = cli.parse_config_args()
    parser = argparse.ArgumentParser(parents=[conf_parser])

    args_default = dict(config.items('Default'))
    args_extraction = dict(config.items('Extraction'))
    args_logmel = dict(config.items('Extraction.Logmel'))
    parser.set_defaults(**args_default, **args_extraction, **args_logmel)

    parser.add_argument('dataset', choices=['training', 'validation', 'test'])
    parser.add_argument('--dataset_dir', type=Path, metavar='DIR')
    parser.add_argument('--extraction_dir', type=Path, metavar='DIR')
    parser.add_argument('--sample_rate', type=int, metavar='RATE')
    parser.add_argument('--n_fft', type=int, metavar='N')
    parser.add_argument('--hop_length', type=int, metavar='N')
    parser.add_argument('--n_mels', type=int, metavar='N')
    parser.add_argument('--overwrite', type=cli.boolean, metavar='BOOL')

    return parser.parse_args(remaining_args)


if __name__ == '__main__':
    sys.exit(extract(parse_args()))
