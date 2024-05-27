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

# initialize so standard_deviation is gain/sqrt(fan_mode) (he initialization)
W1 = torch.randn((n_embed * block_size, n_hidden), generator=g) * (5/3)/((n_embed*block_size) ** 0.5)
# b1 = torch.rand(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * (5/3)/(n_hidden** 0.5)
b2 = torch.randn(vocab_size, generator=g) * 0

bngain = torch.ones((1, n_hidden)) # batch_normalization gain
bnbias = torch.zeros((1, n_hidden)) # batch_normalization bias

bnmean_running = torch.zeros((1, n_hidden)) 
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]

for p in parameters:
    p.requires_grad = True

max_steps = 100_000
batch_size = 32

for i in range(max_steps):
    # minibatch construct
    idx = torch.randint(low=0, high=X_train.shape[0], size=(batch_size, ))

    # forward pass
    embed = C[X_train[idx]]
    embcat = embed.view(embed.shape[0], embed.shape[1] * embed.shape[2])
    hpreact = embcat @ W1 #+ b1
    # normalize hpreact (for batch normalization)
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True) 
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

    with torch.no_grad():
        # calculate mean and std of training set on the go
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani        
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y_train[idx])
    
    # backwards pass
    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 100_000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad


@torch.no_grad()
def split_loss(split):
    X,y = {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test)
    }[split]
    embed = C[X]
    embcat = embed.view(embed.shape[0], embed.shape[1] * embed.shape[2])
    hpreact = embcat @ W1 #+ b1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(f"{split}_loss: ", loss.item())

split_loss('train')
split_loss('valid')


# Generate samples
g = torch.Generator().manual_seed(23490932421)
print("-------------------")
print("Examples of names: ")
for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        # forward pass
        embed = C[torch.tensor([context])]
        h = torch.tanh(embed.view(embed.shape[0], embed.shape[1] * embed.shape[2]) @ W1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        idx = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [idx]
        out.append(idx)
        if idx == 0:
            break
    print(''.join(itos[i] for i in out))