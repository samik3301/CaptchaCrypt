import os
import glob
import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import engine

from model import CaptchaModel
from pprint import pprint


def decode_predictions(preds,encoder):
    preds = preds.permute(1,0,2)  #bs, timestamps, predictions
    preds = torch.softmax(preds,2)
    preds = torch.argmax(preds,2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j,:]:
            k = k-1
            if k == -1:
                temp.append("-")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        tp = "".join(temp)
        cap_preds.append(tp)
    return cap_preds



def run_training():
    # image_files = glob.glob(config.DATA_DIR + "*.png")  #scanning through all the files within a folder
    image_files = glob.glob("D:/Python/CaptchaCrypt/data/vtop_captchas/*")  #scanning through all the files within a folder
    #/../..abcde.png -> [['a','b','c','d','e']]
    targets_org = [x.split("\\")[-1][:-4] for x in image_files]
    #abcde - > [a,b,c,d,e]
    targets = [[c for c in x] for x in targets_org]
    targets_flat = [c for clist in targets for c in clist]  #flattening the list
    #converting a multi dimensional list to a single dimensional list before encoding it 

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(targets_flat)

    target_encoded = [label_encoder.transform(x) for x in targets]
    target_encoded = np.array(target_encoded)+1
    #print(target_encoded)
    #print("\n")
    #print(len(label_encoder.classes_))  #we have 19 different label encoded classes
    #print(targets)
    #print("\n")
    #print(np.unique(targets_flat))


    #NOW SPLITTING THE DATA
    train_images, val_images, train_targets, val_targets, train_orig_targets, test_orig_targets = model_selection.train_test_split(
        image_files,
        target_encoded,
        targets_org,
        test_size=0.1,
        random_state= 42
    )

    train_dataset = dataset.Classification(
        image_paths = train_images, 
        targets = train_targets, 
        resize=(config.IMAGE_HEIGHT,config.IMAGE_WIDTH)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True
    )

    val_dataset = dataset.Classification(
        image_paths = val_images, 
        targets = val_targets, 
        resize=(config.IMAGE_HEIGHT,config.IMAGE_WIDTH)
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = False
    )

    model = CaptchaModel(num_chars=len(label_encoder.classes_))
    model.to(torch.device(config.DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,factor=0.8,patience=5,verbose=True
    )
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds,valid_loss  = engine.eval_fn(model, val_loader)
        valid_cap_preds  = []
        for vp in valid_preds:
            current_preds = decode_predictions(vp,label_encoder)
            valid_cap_preds.extend(current_preds)
        pprint(list(zip(test_orig_targets, valid_cap_preds))[6:11]) #to test change the list also 
        print(f"Epoch: {epoch}, train_loss={train_loss}, valid_loss={valid_loss}")  

    torch.save(model.state_dict(), 'D:/Python/CaptchaCrypt/model_saves/vtop_model.bin') # From here one save model.state_dict NIBA not the entire model object

if __name__ == "__main__":
    run_training()
    #with every step we are predicting 75 values: 
    #the unknown label is not 0 , lets say it is ! or the smiley "!11123343544rf!!!!!!EF" and the size of it will be 75
    #now we have to remove the unknown prediction and count unique occurences
