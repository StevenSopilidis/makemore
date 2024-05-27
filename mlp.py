import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("./data.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
block_size = 3 # how many characters we take into account to predict the next one
vocab_size = len(itos)

def build_dataset(words):
    X, y = [], []
    
    for w in words:
        context = [0] * block_size 
        for ch in w + ".":
            idx = stoi[ch]
            X.append(context)
            y.append(idx)
            context = context[1:] + [idx]
    
    X = torch.tensor(X)
    y = torch.tensor(y)
    return X, y

# split data into train,valid,test
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
X_train, y_train = build_dataset(words[:n1])
X_valid, y_valid = build_dataset(words[n1:n2])
X_test, y_test = build_dataset(words[n2:])


# create embeding table
n_embed = 10
n_hidden = 200

g = torch.Generator().manual_seed(32490123412)
C = torch.randn((vocab_size, n_embed), generator=g)

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        g = torch.Generator().manual_seed(32490123412)
        self.weights = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, X):
        self.out = X @ self.weights
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weights] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.1) -> None:
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # params
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (for running momentum update)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, X):
        if self.training:
            X_mean = X.mean(0, keepdim=True) # batch mean
            X_var = X.var(0, keepdim=True) # batch variance
        else:
            X_mean = self.running_mean
            X_var = self.running_var
        X_normalized = (X - X_mean) / torch.sqrt(X_var + self.eps)
        self.out = self.gamma * X_normalized + self.beta
        # update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * X_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * X_var

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, X):
        self.out = torch.tanh(X)
        return self.out
    
    def parameters(self):
        return []
    
layers = [
    Linear(n_embed * block_size, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size), BatchNorm1D(vocab_size)
]

with torch.no_grad():
    layers[-1].gamma *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weights *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True

max_steps = 10000
batch_size = 32
lossi = []

for i in range(max_steps):
    idx = torch.randint(0, X_train.shape[0], (batch_size, ), generator=g)
    X_batch, y_batch = X_train[idx], y_train[idx]

    # ---- forward pass ----
    embed = C[X_batch]
    X = embed.view(embed.shape[0], -1)
    for layer in layers:
        X = layer(X)
    loss = F.cross_entropy(X, y_batch)

    # ---- backwards pass ----
    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())


# visualize histograms for activations
plt.figure(figsize=(20,4))
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i}, ({layer.__class__.__name__})")
plt.legend(legends)
plt.title("Activation distribution")
plt.show()

# visualize histograms of gradients
plt.figure(figsize=(20,4))
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i}, ({layer.__class__.__name__})")
plt.legend(legends)
plt.title("Activation distribution")
plt.show()