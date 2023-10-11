import torch
import torchvision
from PIL import Image
import glob
from sklearn import preprocessing

# import config
from model import CaptchaModel
from train import decode_predictions


image_files = glob.glob("D:/Python/CaptchaCrypt/data/vtop_captchas/*")  #scanning through all the files within a folder
#/../..abcde.png -> [['a','b','c','d','e']]
targets_org = [x.split("\\")[-1][:-4] for x in image_files]
#abcde - > [a,b,c,d,e]
targets = [[c for c in x] for x in targets_org]
targets_flat = [c for clist in targets for c in clist]  #flattening the list
#converting a multi dimensional list to a single dimensional list before encoding it 
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(targets_flat)

model = CaptchaModel(len(label_encoder.classes_))

model.load_state_dict(torch.load("../model_saves/vtop_model.bin"))

test_img = Image.open("../data/vtop_captchas/2JPCNM.png")
test_tensor = torchvision.transforms.functional.to_tensor(test_img)
# print(test_tensor.shape)
test_tensor = torch.unsqueeze(test_tensor, 0)

with torch.no_grad():
    res = model(test_tensor)
    # print(res[0].shape)
    text_predicted = decode_predictions(res[0], label_encoder)

print(text_predicted)