import torch
import torch.nn.functional as F

words = open("./data.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# create the dataset 
xs, ys = [], []

for w in words:
    wrapper = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(wrapper, wrapper[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        xs.append(idx1)
        ys.append(idx2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
g = torch.Generator().manual_seed(324123214234213)
W = torch.randn((27, 27), generator=g, requires_grad=True)
learning_rate = 10
epochs = 100
loss = 0
a = 0.01

for k in range(epochs):
    x_encoded = F.one_hot(xs, num_classes=27).float()

    # forward pass
    logits = x_encoded @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True) # softmax (y_pred)

    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()

    # backward pass
    W.grad = None # init gradients
    loss.backward()

    # update weights
    W.data += -learning_rate * W.grad

# print some names
g  = torch.Generator().manual_seed(2342341234231412)
for i in range(10):
    out = []
    idx = 0

    while True:
        x_encoded = F.one_hot(torch.tensor([idx]), num_classes=27).float()
        logits = x_encoded @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True) # propabilities for next character

        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        # break character
        if idx == 0: 
            break
    
    print(''.join(out))