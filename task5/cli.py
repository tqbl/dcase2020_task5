import argparse
import configparser

from core.mask import FrameMask


def parse_config_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('-f', '--config_file', metavar='FILE')
    args, remaining_args = conf_parser.parse_known_args()

    # Parse the config file(s). The default config file is a fallback
    # for options that are not specified by the user.
    config = configparser.ConfigParser()
    try:
        config.read_file(open('default.conf'))
        if args.config_file:
            config.read_file(open(args.config_file))
    except FileNotFoundError:
        raise FileNotFoundError(f'Config file not found: {args.config_file}')

    return config, conf_parser, remaining_args


def boolean(arg):
    if arg.lower() == 'true':
        return True
    if arg.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('boolean value expected')


def array(arg):
    return arg.replace(' ', '').split(',')


def mask(arg):
    try:
        return FrameMask(arg)
    except ValueError as error:
        raise argparse.ArgumentTypeError(error)
