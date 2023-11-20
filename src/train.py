import torch
import glob
import numpy as np
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn import model_selection
from matplotlib import pyplot as plt
from pprint import pprint

import config
from model import CaptchaLoss, CaptchaModel
from engine import train_epoch, evaluate
from dataset import DataWrapper

def decode_predictions(preds, encoder):
    # pred - (bs X timesteps X vocabsize) -> (bs X time_steps) -> (bs X 6)
    bs = preds.size(0)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy() # (bs x timesteps)
    decoded_batch = []
    for i in range(bs):
        decoded = []
        for token in preds[i, :]:
            if token == 0:
                decoded.append('-')
            else:
                decoded.append(encoder.inverse_transform([token - 1])[0])
        decoded_str = "".join(decoded)
        decoded_batch.append(decoded_str)
    return decoded_batch

def plot_lossgraph(train_values, val_values, n_epochs, modelname):
    eps = np.arange(1, n_epochs+1)
    plt.plot(eps, train_values, label='Training Loss')
    plt.plot(eps, val_values, label='Validation Loss')
    
    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Set the tick locations
    plt.xticks(np.arange(0, n_epochs+1, 25))
    
    # Display the plot
    plt.legend(loc='best')
    
    plt.savefig(config.PLOTS_SAVE_PATH + modelname + "_loss.jpg")

def train(model, train_loader, val_loader, loss_func, optimizer, scheduler, label_encoder, actual_captchas):

    dev = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    train_losses = []
    val_losses = []
    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, loss_func, optimizer, dev)
        val_preds, val_loss  = evaluate(model, val_loader, loss_func, dev)
        scheduler.step(val_loss)

        captcha_predicted  = []
        for pred in val_preds:
            pred_decoded = decode_predictions(pred, label_encoder)
            captcha_predicted.extend(pred_decoded)
        pprint(list(zip(actual_captchas, captcha_predicted))[6:11])
        print(f"Epoch: {epoch}, train_loss={train_loss}, valid_loss={val_loss}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    plot_lossgraph(np.array(train_losses), np.array(val_losses), n_epochs=config.EPOCHS, modelname="vtopnew")

if __name__ == "__main__":
    # Loading Labels
    image_files = glob.glob(config.DATA_DIR + "*.png")
    if len(image_files) == 0:
        print("ERROR reading images at path, Check path ..")
        exit(0)
    
    targets_org = [x.split("\\")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_org]

    targets_flat = [c for clist in targets for c in clist]
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(targets_flat)

    target_encoded = [label_encoder.transform(x) for x in targets]
    target_encoded = np.array(target_encoded)+1

    print(f"Loaded {len(image_files)} images and labels..\nFound {len(label_encoder.classes_)} different characters in vocabulary")
    # Loading images : we dont need train_original_targets, we just are comparing with the original targets at validation time
    train_images, val_images, train_targets, val_targets, _, val_original_targets = model_selection.train_test_split(
        image_files,
        target_encoded,
        targets_org,
        test_size=0.1,
        random_state= 42
    )
    train_dataset = DataWrapper(paths = train_images, labels = train_targets, resize=(config.IMAGE_HEIGHT,config.IMAGE_WIDTH))
    train_loader  = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=False)
    val_dataset   = DataWrapper(paths = val_images, labels = val_targets, resize=(config.IMAGE_HEIGHT,config.IMAGE_WIDTH))
    val_loader    = DataLoader(val_dataset, config.BATCH_SIZE, shuffle=False)

    # Model, loss func, opt
    model = CaptchaModel(vocabulary_size = len(label_encoder.classes_))
    model.to(torch.device(config.DEVICE if torch.cuda.is_available() else "cpu"))
    loss = CaptchaLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9, patience=15, verbose=True
    )
    # Train and save
    print("Initiate Training ...\n\n")
    train(model, train_loader, val_loader, loss, optimizer, scheduler, label_encoder, val_original_targets)
    torch.save(model.state_dict(), config.MODEL_PATH)
    
