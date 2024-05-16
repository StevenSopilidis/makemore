import torch

words = open("./data.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} # a starts at 1 and so on
stoi['.'] = 0 # "." special char defines start and end of word
itos = {i:s for s,i in stoi.items()}


# array that represents the counts of the bigrams
# row -> first letter
# column -> second letter
# so for example item[i,j] tells us how many items char_i was followed by char_j
# 28 -> 26 letters + our 2 wrappers
N = torch.zeros((27,27), dtype=torch.int32)

for w in words:
    wrapper = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(wrapper, wrapper[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]

        N[idx1, idx2] += 1

P = (N+1).float() # use (N+1) to perform model smoothing so we have not 0 to prob matrix P
P /= P.sum(1, keepdim=True)

g  = torch.Generator().manual_seed(2342341234231412)
for i in range(10):
    out = []
    idx = 0

    while True:
        p = P[idx]

        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        # break character
        if idx == 0: 
            break
    
    print(''.join(out))


log_likelihood = 0.0
n = 0
for w in words:
    wrapper = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(wrapper, wrapper[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]

        prob = P[idx1, idx2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n+=1

nll = -log_likelihood
print(f"{nll=}")
print(f"{nll/n=}")
