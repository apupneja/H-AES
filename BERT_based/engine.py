from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn


def set_values(d, device):
    ids = d["ids"]
    mask = d["mask"]
    scores = d["scores"]

    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    scores = scores.to(device, dtype=torch.float)

    return ids, mask, scores


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids, mask, scores = set_values(d, device)
        optimizer.zero_grad()

        outputs = model(ids=ids, mask=mask)

        loss = nn.MSELoss()
        loss = loss(outputs.reshape(-1), scores)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids, mask, scores = set_values(d, device)
            outputs = model(ids=ids, mask=mask)

            fin_targets.extend(scores.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.reshape(outputs, (-1,)).cpu().detach().numpy().tolist())

    fin_outputs = list(np.around(np.array(fin_outputs), 1))
    fin_targets = list(np.around(np.array(fin_targets), 1))
    return fin_outputs, fin_targets
