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
g = torch.Generator().manual_seed(32490123412)
C = torch.randn((27, 10), generator=g)

W1 = torch.randn((30, 200), generator=g)
W2 = torch.randn((200, 27), generator=g)
b1 = torch.rand(200, generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre # learning rates to search over

lri = []
lossi = []
stepi = []

for i in range(30000):
    # minibatch construct
    idx = torch.randint(low=0, high=X_train.shape[0], size=(32, ))

    # forward pass
    embed = C[X_train[idx]]
    h = torch.tanh(embed.view(embed.shape[0], embed.shape[1] * embed.shape[2]) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y_train[idx])
    # backwards pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # lr = lrs[i]
    lr = 0.1
    for p in parameters:
        p.data += -0.1 * p.grad
    # print(f"---> Epoch: {i}, Loss: {loss}")

    # track stats
    # lri.append(lre[i])
    # lossi.append(loss.log10().item())
    # stepi.append(i)

# for finding the ideal learning_rate
# plt.plot(lri, lossi)
# plt.show()
    
# evaluated valid loss
embed = C[X_valid]
h = torch.tanh(embed.view(embed.shape[0], embed.shape[1] * embed.shape[2]) @ W1 + b1)
logits = h @ W2 + b2
valid_loss = F.cross_entropy(logits, y_valid)
# print(f"Validation loss: {valid_loss}")


# Generate samples
g = torch.Generator().manual_seed(23490932421)

for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        # forward pass
        embed = C[torch.tensor([context])]
        h = torch.tanh(embed.view(embed.shape[0], embed.shape[1] * embed.shape[2]) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        idx = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [idx]
        out.append(idx)
        if idx == 0:
            break
    print(''.join(itos[i] for i in out))