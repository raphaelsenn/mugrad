import numpy as np
from typing import List, Tuple, Optional, Union


class Tensor:
    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray): ValueError(f'{data} is not a np.ndarray')
        self.data = data
        self.grad = None

        self._ctx = None

    def __repr__(self) -> str:
        return f'Tensor(data={self.data}, grad={self.grad})'

    def __add__(self, other):
        return Add.apply(self, other)

    def __neg__(self):
        return Neg.apply(self)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __matmul__(self, other):
        return MatMul.apply(self, other)
    
    def __pow__(self, other):
        return Pow.apply(self, exponent=other)

    def sum(self):
        return Sum.apply(self)

    def mean(self):
        return Mean.apply(self)

    def log(self):
        return Log.apply(self)

    def tanh(self):
        return Tanh.apply(self)

    def relu(self):
        return ReLU.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def backward(self) -> None:
        # TODO: This backpropagation engine needs to be improved!!! 
        if self._ctx is None:
            return None

        if self.grad is None: 
            self.grad = np.ones_like(self.data)
        
        grads = self._ctx.operation.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1: grads = [grads]
        for t, g in zip(self._ctx.parents, grads):
            if t.grad is None:
                if t.data.ndim == 1: t.grad = g.sum(axis=0)
                else: t.grad = g
            else: t.grad += g
            t.backward()


class Context:
    def __init__(self):
        self.saved_tensors = []
        self.parents = []
        self.operation = []
        self.exponent = None

    def save_for_backward(self, *tensors: np.ndarray) -> None:
        self.saved_tensors.extend(tensors)


class Function:
    @classmethod
    def apply(cls, *inputs: Tensor, exponent=None) -> Tensor:
        ctx = Context()
        ctx.parents = list(inputs)
        ctx.operation = cls
        ctx.exponent = exponent

        input_data = [t.data for t in inputs]
        output_data = cls.forward(ctx, *input_data) if exponent is None else cls.forward(ctx, *input_data, exponent)

        out_tensor = Tensor(output_data)
        out_tensor._ctx = ctx
        return out_tensor

    @staticmethod
    def forward(ctx: Context, *args: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError()


class Add(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x, y)
        return x + y
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x, y)
        return x * y
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = ctx.saved_tensors
        return grad_output * y, grad_output * x


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x, y)
        return x @ y
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = ctx.saved_tensors
        return grad_output @ y.T, x.T @ grad_output


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, input: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(-1.0*input)
        return -1.0 * input
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input, = ctx.saved_tensors 
        return grad_output * (-1.0 * np.ones_like(input))


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, input: np.ndarray, exponent: float) -> np.ndarray:
        ctx.exponent = exponent 
        ctx.save_for_backward(input)
        return input**exponent
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        input, = ctx.saved_tensors
        exponent = ctx.exponent
        return grad_output * (exponent * (input**(exponent-1)))


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, input: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(input)
        return np.sum(input)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)


class Mean(Function):
    @staticmethod
    def forward(ctx: Context, input: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(input)
        return np.mean(input)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        input, = ctx.saved_tensors
        return grad_output * ((1.0/input.shape[0]) * np.ones_like(input))


class Log(Function):
    @staticmethod
    def forward(ctx: Context, input: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(input)
        return np.log(input)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        input, = ctx.saved_tensors
        return grad_output * (1.0 / (input+1e-8))


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, input: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(input)
        relu = np.maximum(0.0, input) 
        return relu

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        input, = ctx.saved_tensors
        return grad_output * (input > 0)


class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, input: np.ndarray) -> np.ndarray:
        tanh = np.tanh(input)
        ctx.save_for_backward(tanh)
        return tanh
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        tanh, = ctx.saved_tensors
        return grad_output * (1.0 - tanh**2)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, input: np.ndarray) -> np.ndarray:
        sigmoid = 1.0 / (1.0 + np.exp(-input)) 
        ctx.save_for_backward(sigmoid)
        return sigmoid
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        sigmoid, = ctx.saved_tensors
        return grad_output * (sigmoid * (1 - sigmoid))