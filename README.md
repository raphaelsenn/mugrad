# mugrad
Yes, yet another autograd engine with PyTorch-like API... It is called mugrad!

**Whats the purpose of mugrad?**
* Just to teach myself (and maybe/hopefully others) how something like PyTorch works (or kind of works.. pytorch is much more complicated than mugrad :D).


## Usage

### Creating Neural Networks with mugrad

```python
from mugrad.mugrad import Tensor
from mugrad.nn import Module, Linear


class MuNeuralNet(Module):
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = x.relu() 
        x = self.fc2(x)
        x = x.relu()
        return x

model = MuNeuralNet()
```


```python
from mugrad.mugrad import Tensor
from mugrad.nn import Module, Linear, ReLU, Sequential


model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
    ReLU()
])
```


### Training Neural Networks with mugrad

```python
import mugrad as mu
from mugrad.nn import CrossEntropyLoss


optimizer = mu.optim.SGD(params=model.parameteres(), lr=0.001)
criterion = CrossEntropyLoss()

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_test)
    loss.backward()
    optimizer.step()
```
