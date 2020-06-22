import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import pytorch.models as models
from pytorch.data import MappedDataLoader, ContextualizedDataset, SimpleDataset
from pytorch.logging import Logger
from pytorch.specaug import SpecAugment


def train(x_train, y_train, x_val, y_val, log_dir, model_dir, **params):
    if params['seed'] >= 0:
        _ensure_reproducibility(params['seed'])

    # Determine which device (GPU or CPU) to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Determine how to load the model and the data
    # Whether STC features are used will affect this
    model_params = {
        'model_type': params['model'],
        'n_classes': y_train.shape[-1],
    }
    if params['use_stc']:
        create_dataset = ContextualizedDataset
        model_params['n_aux'] = x_train[1].shape[1]
    else:
        create_dataset = SimpleDataset

    # Instantiate neural network
    model = models.create_model(**model_params).to(device)
    # Use cross-entropy loss and Adam optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(model.optimizer_parameters(lr=params['lr']))
    # Use StepLR scheduler to decay learning rate regularly
    scheduler = StepLR(
        optimizer,
        step_size=params['lr_decay_rate'],
        gamma=params['lr_decay'],
    )

    # Load training state from last checkpoint if applicable
    if not params['overwrite']:
        initial_epoch = _load_checkpoint(
            model, optimizer, scheduler, model_dir)
        if initial_epoch >= params['n_epochs']:
            return
    else:
        initial_epoch = 0

    # Use helper classes to iterate over data in batches
    batch_size = params['batch_size']
    loader_train = MappedDataLoader(
        create_dataset(x_train, y_train.values),
        device=device,
        batch_size=batch_size,
        shuffle=True,
    )
    loader_val = MappedDataLoader(
        create_dataset(x_val, y_val.values),
        device=device,
        batch_size=batch_size,
    )

    # Instantiate Logger to record training/validation performance and
    # save checkpoint to disk after every epoch.
    logger = Logger(log_dir, model_dir, params['overwrite'])

    for epoch in range(initial_epoch, params['n_epochs']):
        # Enable data augmentation after 5 epochs
        if epoch == 5 and params['augment']:
            loader_train.dataset.transform = SpecAugment()

        # Train model using training set
        pbar = tqdm(loader_train)
        pbar.set_description(f'Epoch {epoch}')
        _train(pbar, model.train(), criterion, optimizer, logger)

        # Evaluate model using validation set
        _validate(loader_val, y_val.index, model.eval(), criterion, logger)

        # Invoke learning rate scheduler
        scheduler.step()

        # Log results and save model to disk
        logger.step(model, optimizer, scheduler)

    logger.close()


def predict(x, epoch, model_dir, use_stc=False, batch_size=128):
    # Determine which device (GPU or CPU) to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model from disk
    model_dir = model_dir / f'model.{epoch:02d}.pth'
    checkpoint = torch.load(model_dir, map_location=device)
    model = models.create_model(**checkpoint['creation_args'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    create_dataset = ContextualizedDataset if use_stc else SimpleDataset
    loader = MappedDataLoader(
        create_dataset(x),
        device=device,
        batch_size=batch_size,
    )
    with torch.no_grad():
        y_pred = torch.cat([model(batch_x).sigmoid().data
                            for batch_x, in loader])
    return y_pred.cpu().numpy()


def _train(data, model, criterion, optimizer, logger):
    for batch_x, batch_y in data:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        logger.log('loss', loss.item())


def _validate(loader, index, model, criterion, logger):
    y_true = loader.dataset.y
    with torch.no_grad():
        y_pred = torch.cat([model(batch_x).sigmoid().data
                            for batch_x, _ in loader])

    loss = criterion(y_pred, y_true)
    logger.log('val_loss', loss.item())


def _load_checkpoint(model, optimizer, scheduler, model_dir):
    # Check model directory for existing checkpoints
    paths = sorted(model_dir.glob('model.[0-9][0-9].pth'))
    if len(paths) == 0:
        return 0

    # Load training state from last checkpoint
    path = sorted(paths)[-1]
    epoch = int(str(path)[-6:-4])
    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'])
    if checkpoint.get('cuda_rng_state') is not None \
            and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    if epoch != checkpoint['epoch']:
        # The epoch to resume from is determined by the file name of the
        # checkpoint file. If this number doesn't agree with the number
        # that is recorded internally, raise an error.
        raise RuntimeError('Epoch mismath')

    return epoch + 1


def _ensure_reproducibility(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
