import torch.nn as nn
from torch.autograd import Variable


class ActivationException(Exception):
    pass


class LinearBlock(nn.Module):

    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.0, activation: str = "lrelu"):
        super(LinearBlock, self).__init__()

        self._linear = nn.Linear(in_features=input_size, out_features=output_size)
        self._dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else None

        if activation == "lrelu":
            self._activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "sigm":
            self._activation = nn.Sigmoid()
        elif activation == "linear":
            self._activation = None
        elif activation == "tanh":
            self._activation = nn.Tanh()
        else:
            raise ActivationException(
                "Invalid activation: 'lrelu', 'tanh', 'sigm' or 'linear' are possible activation functions")

    def forward(self, x: Variable):

        out = self._linear(x)

        if self._dropout:
            out = self._dropout(out)

        if self._activation:
            out = self._activation(out)

        return out


class Discriminator(nn.Module):

    def __init__(self, input_size: int = 784, output_size: int = 1):
        super(Discriminator, self).__init__()

        self._input_size = input_size
        self._output_size = output_size

        self._layers = nn.Sequential(
            LinearBlock(input_size=self._input_size, output_size=1024, dropout_rate=0.3),
            LinearBlock(input_size=1024, output_size=512, dropout_rate=0.3),
            LinearBlock(input_size=512, output_size=256, dropout_rate=0.3),
            LinearBlock(input_size=256, output_size=self._output_size, activation='sigm')
        )

    def forward(self, x: Variable):
        return self._layers.forward(x)

    @property
    def input_dim(self):
        return self._input_size

    @property
    def output_dim(self):
        return self._output_size


class Generator(nn.Module):

    def __init__(self, input_size: int = 100, output_size: int = 784):
        super(Generator, self).__init__()
        self._input_size = input_size
        self._output_size = output_size

        self._layers = nn.Sequential(
            LinearBlock(input_size=self._input_size, output_size=256, dropout_rate=0.2),
            LinearBlock(input_size=256, output_size=512, dropout_rate=0.2),
            LinearBlock(input_size=512, output_size=1024, dropout_rate=0.2),
            LinearBlock(input_size=1024, output_size=self._output_size, activation='tanh')
        )

    def forward(self, x: Variable):
        return self._layers.forward(x)

    @property
    def input_dim(self):
        return self._input_size

    @property
    def output_dim(self):
        return self._output_size
