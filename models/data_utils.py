import torch

def batchify(inp, batch_size, device):
    seq = []
    q = []
    target = []
    for el in inp:
        seq.append(el[0].to(device))
        q.append(el[1].to(device))
        target.append(el[2].to(device))
    return torch.stack(seq), torch.stack(q), torch.stack(target)
