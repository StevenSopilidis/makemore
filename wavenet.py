import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("./data.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
block_size = 8 # how many characters we take into account to predict the next one
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
n_embed = 24
n_hidden = 128

g = torch.Generator().manual_seed(32490123412)

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
            if X.ndim == 2:
                dim = 0 
            elif X.ndim == 3:
                dim = (0, 1)
            X_mean = X.mean(dim, keepdim=True) # batch mean
            X_var = X.var(dim, keepdim=True) # batch variance
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
    

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weights = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, idx):
        self.out = self.weights[idx]
        return self.out
    
    def parameters(self):
        return [self.weights]
    
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
        
    def __call__(self, X):
        B, T, C = X.shape
        X = X.view(B, T//self.n, C*self.n)
        
        if X.shape[1] == 1:
            X = X.squeeze(1)
        
        self.out = X
        return self.out
    
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        self.out = X
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

torch.manual_seed(1242334223)
    
model = Sequential([
    Embedding(vocab_size, n_embed),
    FlattenConsecutive(2), Linear(n_embed * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

with torch.no_grad():
    model.layers[-1].weights *= 0.1


max_steps = 10000
batch_size = 32
lossi = []

parameters = [p for layer in model.layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True


for i in range(max_steps):
    idx = torch.randint(0, X_train.shape[0], (batch_size, ), generator=g)
    X_batch, y_batch = X_train[idx], y_train[idx]

    # ---- forward pass ----
    logits = model(X_batch)
    loss = F.cross_entropy(logits, y_batch)

    # ---- backwards pass ----
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


plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.show()


# put model into eval mode (needed for batchnorm especially)
for layer in model.layers:
  layer.training = False

# evaluate the loss
@torch.no_grad() # this decorator disables gradient tracking inside pytorch
def split_loss(split):
    x,y = {
    'train': (X_train, y_train),
    'val': (X_valid, y_valid),
    'test': (X_test, y_test),
    }[split]

    logits = model(X_batch)
    loss = F.cross_entropy(logits, y_batch)
    print(split, loss.item())

split_loss('train')
split_loss('val')