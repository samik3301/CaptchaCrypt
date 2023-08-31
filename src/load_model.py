import torch
import config

model = torch.load(config.MODEL_PATH)
model.eval()
