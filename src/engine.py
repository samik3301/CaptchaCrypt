import torch
import numpy as np
from tqdm import tqdm


def train_epoch(model, data_loader, loss_func, optimizer, device):
    model.train()
    loss_graph = []
    for img_batch, target_batch in tqdm(data_loader):
        img_batch  = img_batch.to(device)
        target_batch = target_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(img_batch)
        loss = loss_func(preds, target_batch)
        loss_graph.append(loss.item())

        loss.backward()
        optimizer.step()
    return np.mean(loss_graph)

def evaluate(model, data_loader, loss_func, device):
    model.eval()
    loss_graph = []
    batch_preds = []
    with torch.no_grad():
        for img_batch, target_batch in tqdm(data_loader):
            img_batch  = img_batch.to(device)
            target_batch = target_batch.to(device)
            
            preds = model(img_batch)
            loss = loss_func(preds, target_batch)
            loss_graph.append(loss.item())
            batch_preds.append(preds)
    return batch_preds, np.mean(loss_graph)
