from typing import Tuple
import torch
from torch import nn as nn
import torch.nn.functional as F

import numpy as np
import pdb

import pytorch_utils as ptu

def identity(x):
    return x

def relativeErrorLossFunction(additiveConstant, reduction="mean"):
    """
    Loss function based on https://arxiv.org/pdf/2105.05249.pdf
    Requires that true+additiveConstant and pred+additiveConstant > 0
    """
    mseFunc = nn.MSELoss(reduction=reduction)
    def lossFunction(pred, true):
        # This is equivalent to mean((ln(true+C)-ln(pred+C))^2)
        return mseFunc(torch.log(true + additiveConstant), torch.log(pred + additiveConstant))
    return lossFunction

## https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch
def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + F.softplus(-2. * x) - np.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class simple_CNN(nn.Module):
    def __init__(
            self,
            input_dim,
            input_channels,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            max_pool_sizes,  # Max pool should be in the form [int, ...], use 1 to indicate no max pool
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_flatten=False,
            mlp_sizes=None):

        assert len(kernel_sizes) == len(n_channels) == len(strides) == len(paddings) == len(max_pool_sizes)
        super().__init__()

        self.input_dim = input_dim
        self.input_channels = input_channels
        self.kernel_sizes = kernel_sizes
        self.n_channels = n_channels
        self.strides = strides
        self.paddings = paddings
        self.max_pool_sizes = max_pool_sizes
        self.max_pools = []
        self.mlp_sizes = mlp_sizes
        for a_size in max_pool_sizes:
            if a_size is None:
                self.max_pools.append(identity)
            else:
                self.max_pools.append(nn.MaxPool2d(a_size))

        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.output_flatten = output_flatten

        self.conv_layers = nn.ModuleList()

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            conv = nn.Conv2d(input_channels, out_channels, kernel_size, stride=stride, padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            self.conv_layers.append(conv)
            input_channels = out_channels

        if mlp_sizes is not None:  # Add mlp modules
            assert output_flatten  # output_flatten needs to be true for mlp to work afterwards
            self.mlp_layers = nn.ModuleList()
            input_size = self.get_output_dim(include_mlp=False)[-1]

            for a_size in mlp_sizes:
                mlp = nn.Linear(input_size, a_size)
                hidden_init(mlp.weight)

                self.mlp_layers.append(mlp)
                input_size = a_size
        else:
            self.mlp_layers = None

        print("CNN dims: {}".format(self.get_output_dim()))

    # Input: input is (B,C,D,D)
    def forward(self, input):
        # for i in range(len(self.conv_layers)):
        #     input = self.hidden_activation(self.conv_layers[i](input))
        #     input = self.max_pools[i](input)
        # for i in range(len(self.conv_layers)):
        #     layer: torch.nn.Conv2d = self.conv_layers[i]
        #     input = self.hidden_activation(layer(input))
        #     max_pool2d: nn.MaxPool2d = self.max_pools[i]
        #     input = max_pool2d(input)
        for layer in self.conv_layers:
            input = self.hidden_activation(layer(input))

        if self.output_flatten:
            input = input.reshape((input.shape[0], -1))  # (B,C*D*D)
        if self.mlp_layers is not None:
            for i, layer in enumerate(self.mlp_layers):
                if i < len(self.mlp_layers)-1: # If not last layer
                    input = self.hidden_activation(layer(input))
                else:  # Don't apply regular hidden_activation
                    input = layer(input)
            # for layer in self.mlp_layers:
            #     input = self.hidden_activation(layer(input))
        input = self.output_activation(input)
        return input

    def get_output_dim(self, include_mlp=True):
        cur_dim = self.input_dim
        all_dims = [cur_dim]
        for i in range(len(self.conv_layers)):
            cur_dim = (cur_dim + 2*self.paddings[i] - (self.kernel_sizes[i]-1))//self.strides[i] + 1
            cur_dim = cur_dim // self.max_pool_sizes[i]
            all_dims.append(cur_dim)
        if self.output_flatten:
            all_dims.append("Flatten")
            all_dims.append(self.n_channels[-1]*cur_dim*cur_dim)
        if include_mlp and self.mlp_sizes is not None:
            all_dims.extend(self.mlp_sizes)
        return all_dims


class simple_MLP(nn.Module):
    def __init__(
            self,
            input_size,
            sizes,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity):
        super().__init__()

        self.input_size = input_size
        self.sizes = sizes
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.mlp_layers = nn.ModuleList()
        self.activations = []
        for a_size in sizes:
            mlp = nn.Linear(input_size, a_size)
            hidden_init(mlp.weight)

            self.mlp_layers.append(mlp)
            input_size = a_size

    def forward(self, input):
        ### This version works with torch.jit.script while others may not
        for i, layer in enumerate(self.mlp_layers):
            if i < len(self.mlp_layers)-1: # If not last layer
                input = self.hidden_activation(layer(input))
            else:
                input = self.output_activation(layer(input))
        # pdb.set_trace()
        # for layer, activation in zip(self.mlp_layers, self.activations):
        #     input = activation(layer(input))
            # input = self.hidden_activation(layer(input))
        # input = self.mlp_layers[-1](input)
        # return self.output_activation(input)
        return input

class ClampClass(nn.Module):
    clampMax: int
    def __init__(self, clampMax: int) -> None:
        super().__init__()
        self.clampMax = clampMax

    def forward(self, x):
        return torch.clamp(x, 0, self.clampMax)


class MultiHeadActivation(nn.Module):
    def __init__(self, clampMax) -> None:
        super().__init__()
        # self.f1, self.f2, self.f3 = nn.ReLU(), identity, nn.Sigmoid()
        self.f1 = ClampClass(clampMax)
        self.f2, self.f3 = identity, identity

    def forward(self, preds):
        return torch.stack([self.f1(preds[:,0]),
                            self.f2(preds[:,1]),
                            self.f3(preds[:,2])], dim=1)

########
class simple_CNN2(nn.Module):
    def __init__(
            self,
            input_dim,
            input_channels,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            max_pool_sizes,  # Max pool should be in the form [int, ...], use 1 to indicate no max pool
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_flatten=False,
            mlp_sizes=None,
            mlp_extra_input_size=0):

        assert len(kernel_sizes) == len(n_channels) == len(strides) == len(paddings) == len(max_pool_sizes)
        super().__init__()

        self.input_dim = input_dim
        self.input_channels = input_channels
        self.kernel_sizes = kernel_sizes
        self.n_channels = n_channels
        self.strides = strides
        self.paddings = paddings
        self.max_pool_sizes = max_pool_sizes
        self.max_pools = []
        self.mlp_sizes = mlp_sizes
        self.mlp_extra_input_size = mlp_extra_input_size
        for a_size in max_pool_sizes:
            if a_size is None:
                self.max_pools.append(identity)
            else:
                self.max_pools.append(nn.MaxPool2d(a_size))

        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.output_flatten = output_flatten

        self.conv_layers = nn.ModuleList()

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            conv = nn.Conv2d(input_channels, out_channels, kernel_size, stride=stride, padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            self.conv_layers.append(conv)
            input_channels = out_channels

        if mlp_sizes is not None:  # Add mlp modules
            assert output_flatten  # output_flatten needs to be true for mlp to work afterwards
            self.mlp_layers = nn.ModuleList()
            input_size = self.get_output_dim(include_mlp=False)[-1]
            input_size += self.mlp_extra_input_size # Add input after the CNN layer

            for a_size in mlp_sizes:
                mlp = nn.Linear(input_size, a_size)
                hidden_init(mlp.weight)

                self.mlp_layers.append(mlp)
                input_size = a_size
        else:
            self.mlp_layers = None

        print("CNN dims: {}".format(self.get_output_dim()))

    def forward(self, input, mlp_extra_input):
        """
        Input: input is (B,C,D,D)
            mlp_extra_input: (B,S)
        """
        # for i in range(len(self.conv_layers)):
        #     input = self.hidden_activation(self.conv_layers[i](input))
        #     input = self.max_pools[i](input)
        # for i in range(len(self.conv_layers)):
        #     layer: torch.nn.Conv2d = self.conv_layers[i]
        #     input = self.hidden_activation(layer(input))
        #     max_pool2d: nn.MaxPool2d = self.max_pools[i]
        #     input = max_pool2d(input)
        for layer in self.conv_layers:
            input = self.hidden_activation(layer(input))

        if self.output_flatten:
            input = input.reshape((input.shape[0], -1))  # (B,C*D*D)
        input = torch.cat([input, mlp_extra_input], dim=1)  # (B,C*D*D+S)
        if self.mlp_layers is not None:
            for i, layer in enumerate(self.mlp_layers):
                if i < len(self.mlp_layers)-1: # If not last layer
                    input = self.hidden_activation(layer(input))
                else:  # Don't apply regular hidden_activation
                    input = layer(input)
            # for layer in self.mlp_layers:
            #     input = self.hidden_activation(layer(input))
        input = self.output_activation(input)
        return input

    def get_output_dim(self, include_mlp=True):
        cur_dim = self.input_dim
        all_dims = [cur_dim]
        for i in range(len(self.conv_layers)):
            cur_dim = (cur_dim + 2*self.paddings[i] - (self.kernel_sizes[i]-1))//self.strides[i] + 1
            cur_dim = cur_dim // self.max_pool_sizes[i]
            all_dims.append(cur_dim)
        if self.output_flatten:
            all_dims.append("Flatten")
            all_dims.append(self.n_channels[-1]*cur_dim*cur_dim)
        if include_mlp and self.mlp_sizes is not None:
            all_dims.extend(self.mlp_sizes)
        return all_dims


class simple_MLP2(nn.Module):
    def __init__(
            self,
            input_size,
            sizes,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=[identity]):
        super().__init__()

        self.input_size = input_size
        self.sizes = sizes
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        if len(output_activation) == 1:
            self.output_activation = output_activation[0]
        else:
            self.output_activation = lambda x: [func(x) for x,func in zip(x, output_activation)]

        self.mlp_layers = nn.ModuleList()
        self.activations = []
        for a_size in sizes:
            mlp = nn.Linear(input_size, a_size)
            hidden_init(mlp.weight)

            self.mlp_layers.append(mlp)
            input_size = a_size

    def forward(self, input):
        ### This version works with torch.jit.script while others may not
        for i, layer in enumerate(self.mlp_layers):
            if i < len(self.mlp_layers)-1: # If not last layer
                input = self.hidden_activation(layer(input))
            else:
                input = self.output_activation(layer(input))
        # pdb.set_trace()
        # for layer, activation in zip(self.mlp_layers, self.activations):
        #     input = activation(layer(input))
            # input = self.hidden_activation(layer(input))
        # input = self.mlp_layers[-1](input)
        # return self.output_activation(input)
        return input
