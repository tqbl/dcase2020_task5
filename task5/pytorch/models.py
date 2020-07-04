import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch.qkcnn as qkcnn


class GCNN(nn.Module):
    def __init__(self, n_classes, n_aux=None):
        super().__init__()

        self.conv_layers = nn.Sequential(
            GatedConvBlock(in_channels=1, out_channels=64),
            GatedConvBlock(in_channels=64, out_channels=128),
            GatedConvBlock(in_channels=128, out_channels=256),
            GatedConvBlock(in_channels=256, out_channels=512),
            GatedConvBlock(in_channels=512, out_channels=512),
        )

        n_aux_out = (n_aux or 0) * 2
        if n_aux is not None:
            self.aux_fc = nn.Linear(n_aux, n_aux_out)
            self.aux_bn = nn.BatchNorm1d(n_aux_out)

        self.classifier = nn.Linear(512 + n_aux_out, n_classes)

    def forward(self, x):
        if isinstance(x, tuple):
            x, aux = x
        else:
            aux = None

        x = self.conv_layers(x)
        x = x.mean(dim=[2, 3]).squeeze()
        if aux is not None:
            aux_out = self.aux_bn(self.aux_fc(aux)).relu()
            x = torch.cat([x, aux_out], dim=1)
        return self.classifier(x)

    def optimizer_parameters(self, lr):
        return [{'params': self.parameters(), 'lr': lr}]


class GatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 pool_size=(2, 2), kernel_size=3, **args):
        super(GatedConvBlock, self).__init__()

        self.conv1 = GatedConv(in_channels, out_channels, kernel_size, **args)
        self.conv2 = GatedConv(out_channels, out_channels, kernel_size, **args)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool_size = pool_size

    def forward(self, x, pool_size=(2, 2)):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        if self.pool_size != (1, 1):
            x = F.max_pool2d(x, pool_size)
        return x


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, **args):
        super(GatedConv, self).__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding, bias=False, **args)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding, bias=False, **args)

    def forward(self, x, pool_size=(2, 2)):
        x1 = self.conv1(x)
        x2 = self.conv2(x).sigmoid()
        return x1 * x2


def create_model(model_type, n_classes, n_aux=None):
    if model_type.lower() == 'gcnn':
        model = GCNN(n_classes, n_aux)
    elif model_type.lower() == 'qkcnn10':
        model = qkcnn.Cnn10(527, n_aux)  # Initially 527-class output
        url = 'https://zenodo.org/record/3576403/files/Cnn10_mAP=0.380.pth'
        checkpoint = torch.hub.load_state_dict_from_url(
            url, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        in_features = model.fc_audioset.in_features + (n_aux or 0) * 2
        model.fc_audioset = nn.Linear(in_features, n_classes)
    else:
        raise ValueError(f'Unrecognized model type: {model_type}')

    # Save the arguments that were passed to create the model
    model.creation_args = {
        'model_type': model_type,
        'n_classes': n_classes,
        'n_aux': n_aux,
    }

    return model
