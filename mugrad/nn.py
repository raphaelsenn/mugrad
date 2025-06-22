import numpy as np
from tensor import Tensor, Function
from typing import Union


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None
            # p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features 
        self.weight = Tensor(np.random.uniform(-1.0/(in_features**0.5), 1.0/(in_features**0.5), size=(in_features, out_features)))
        self.bias = Tensor(np.random.uniform(-1.0/(in_features**0.5), 1.0/(in_features**0.5), size=(out_features,)))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]
    

class TanH(Module):
    def forward(self, x):
        return x.tanh()


class BCELoss(Module):
    def forward(self, input, target, reduction:str='mean'):
        # return -(target * input.log() + (1 - target) * (1 - input).log())
        pass

class Sequential(Module):
    def __init__(self, *layers: Union[Tensor, Function]) -> None:
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        return [layer.parameters() for layer in self.layers]