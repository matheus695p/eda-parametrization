import torch.nn as nn
# import torch.nn.functional as F


class RegNeuralNet(nn.Module):
    """
    Red neuronal feed fordwar fully connected

    ...
    Attributes
    ----------
    object : herencia
        módulo de redes neuronales de pytorch

    Methods
    -------
    forward(x):
        propagar hacia adelante la red.
    """

    def __init__(self, num_feature, num_targets):
        super(RegNeuralNet, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_targets)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        # W1x + b1
        x = self.layer_1(x)
        # BatchNormalization
        x = self.batchnorm1(x)
        # Activation
        x = self.relu(x)
        # Dropout
        x = self.dropout(x)
        # W2x + b2
        x = self.layer_2(x)
        # BatchNormalization
        x = self.batchnorm2(x)
        # Activation
        x = self.relu(x)
        # Dropout
        x = self.dropout(x)
        # W3x + b3
        x = self.layer_3(x)
        # BatchNormalization
        x = self.batchnorm3(x)
        # Activation
        x = self.relu(x)
        # Dropout
        x = self.dropout(x)
        # W4x + b4
        x = self.layer_out(x)

        return x
