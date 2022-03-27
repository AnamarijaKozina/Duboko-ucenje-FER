import torch
import torch.nn as nn
import torch.optim as optim


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2, 3.2, -1])
Y = torch.tensor([3, 5, 7.4, -1])

n = len(Y)

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1)

for i in range(100):
    # afin regresijski model
    Y_ = a*X + b

    diff = (Y-Y_)

    # kvadratni gubitak
    loss = torch.sum(diff**2)/n

    # računanje gradijenata
    loss.backward()

    # korak optimizacije
    optimizer.step()

    if(i%20==0):
      print(f'Gradijenti koje racuna PyTorch: dL/da = {a.grad.item()}, dL/db = {b.grad.item()}')
      print(f'Gradijenti dobiveni analiticki: dL/da = {-2*sum(diff * X)/n}, dL/db = {-2*sum(diff)/n}')

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()
    if(i%20==0):
      print(f'step: {i}, loss:{loss:.8f}, Y_:{Y_.tolist()}, a:{a.item():.8f}, b: {b.item():.8f}\n')