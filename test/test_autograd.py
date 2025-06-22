from mugrad.tensor import Tensor
import numpy as np
import torch
import pytest

@pytest.fixture(autouse=True)
def set_numpy_seed():
    np.random.seed(42)


def test_add_backward():
    x_np = np.random.randn(3, 3)
    y_np = np.random.randn(3, 3)

    # mugrad
    x = Tensor(x_np)
    y = Tensor(y_np)
    z = (x + y).sum()
    z.backward()

    # pytorch for reference
    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = torch.tensor(y_np, requires_grad=True)
    z_torch = (x_torch + y_torch).sum()
    z_torch.backward()

    assert np.allclose(x.grad, x_torch.grad.numpy())
    assert np.allclose(y.grad, y_torch.grad.numpy())


def test_mul_backward():
    x_np = np.random.randn(3, 3)
    y_np = np.random.randn(3, 3)

    # mugrad
    x = Tensor(x_np)
    y = Tensor(y_np)
    z = (x * y).sum()
    z.backward()

    # pytorch for reference
    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = torch.tensor(y_np, requires_grad=True)
    z_torch = (x_torch * y_torch).sum()
    z_torch.backward()

    assert np.allclose(x.grad, x_torch.grad.numpy())
    assert np.allclose(y.grad, y_torch.grad.numpy())


def test_matmul_backward():
    x_np = np.random.randn(3, 3)
    y_np = np.random.randn(3, 3)

    # mugrad
    x = Tensor(x_np)
    y = Tensor(y_np)
    z = (x @ y).sum()
    z.backward()

    # pytorch for reference
    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = torch.tensor(y_np, requires_grad=True)
    z_torch = (x_torch @ y_torch).sum()
    z_torch.backward()

    assert np.allclose(x.grad, x_torch.grad.numpy())
    assert np.allclose(y.grad, y_torch.grad.numpy())


def test_sub_backward():
    x_np = np.random.randn(3, 3)
    y_np = np.random.randn(3, 3)

    # mugrad
    x = Tensor(x_np)
    y = Tensor(y_np)
    z = (x - y).sum()
    z.backward()

    # pytorch for reference
    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = torch.tensor(y_np, requires_grad=True)
    z_torch = (x_torch - y_torch).sum()
    z_torch.backward()

    assert np.allclose(x.grad, x_torch.grad.numpy())
    assert np.allclose(y.grad, y_torch.grad.numpy())



def test_tanh_backward():
    x_np = np.random.randn(1000, 784)
    w_np = np.random.randn(784, 10)
    b_np = np.random.randn(10)

    # mugrad
    x = Tensor(x_np)
    w = Tensor(w_np)
    b = Tensor(b_np)

    z = (x @ w + b).tanh().sum()
    z.backward()

    # pytorch for reference
    x_torch = torch.tensor(x_np, requires_grad=True)
    w_torch = torch.tensor(w_np, requires_grad=True)
    b_torch = torch.tensor(b_np, requires_grad=True)
    z_torch = (x_torch @ w_torch + b_torch).tanh().sum()
    z_torch.backward()

    print(b.data, b.grad.sum(axis=0))
    print(b_torch.data, b_torch.grad)
    
    assert np.allclose(x.grad, x_torch.grad.numpy())
    assert np.allclose(w.grad, w_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_sigmoid_backward():
    x_np = np.random.randn(1000, 784)
    w_np = np.random.randn(784, 10)
    b_np = np.random.randn(10)

    # mugrad
    x = Tensor(x_np)
    w = Tensor(w_np)
    b = Tensor(b_np)

    z = (x @ w + b).sigmoid().sum()
    z.backward()

    # pytorch for reference
    x_torch = torch.tensor(x_np, requires_grad=True)
    w_torch = torch.tensor(w_np, requires_grad=True)
    b_torch = torch.tensor(b_np, requires_grad=True)
    z_torch = (x_torch @ w_torch + b_torch).sigmoid().sum()
    z_torch.backward()

    print(b.data, b.grad.sum(axis=0))
    print(b_torch.data, b_torch.grad)
    
    assert np.allclose(x.grad, x_torch.grad.numpy())
    assert np.allclose(w.grad, w_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())

def test_relu_backward():
    x_np = np.random.randn(1000, 784)
    w_np = np.random.randn(784, 10)
    b_np = np.random.randn(10)

    # mugrad
    x = Tensor(x_np)
    w = Tensor(w_np)
    b = Tensor(b_np)

    z = (x @ w + b).relu().sum()
    z.backward()

    # pytorch for reference
    x_torch = torch.tensor(x_np, requires_grad=True)
    w_torch = torch.tensor(w_np, requires_grad=True)
    b_torch = torch.tensor(b_np, requires_grad=True)
    z_torch = (x_torch @ w_torch + b_torch).relu().sum()
    z_torch.backward()

    print(b.data, b.grad.sum(axis=0))
    print(b_torch.data, b_torch.grad)
    
    assert np.allclose(x.grad, x_torch.grad.numpy())
    assert np.allclose(w.grad, w_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())