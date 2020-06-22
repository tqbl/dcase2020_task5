import json
import time


def log_parameters(output_path, **params):
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)


def load_data(subset, data_loader):
    n = len(subset.tags.index.unique())
    print(f'Loading {subset.name} set ({n} instances)...')
    return timeit(lambda: data_loader(subset),
                  f'Loaded {subset.name} set')


def timeit(callback, message):
    start = time.time()
    x = callback()

    print('{} in {:.3f} seconds'.format(message, time.time() - start))

    return x
