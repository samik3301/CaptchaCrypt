import model
import torch
import engine
import os
import config

PATH = r'/Users/samik/Desktop/Programming/CaptchaCrypt/src/model_save/model.pt'
model = torch.load(PATH)
model.eval()
