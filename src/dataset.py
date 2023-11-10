import albumentations
import torch
from torch.utils.data import Dataset
import numpy as np


from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataWrapper(Dataset):
    def __init__(self, paths, labels, resize=None):
        self.paths = paths
        self.targets = labels
        self.resize = resize
        # self.aug = albumentations.Compose(
        #     [
        #         albumentations.Normalize(always_apply=True)
        #     ]
        # )

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("L")
        target = self.targets[idx]
        if self.resize != None:
            image = image.resize((self.resize[1], self.resize[0]),resample = Image.BILINEAR)
        image = np.array(image)  
        # augmented = self.aug(image = image)
        # image = augmented["image"] # normalizing timage via aug
        image = np.transpose(image, (1, 0)).astype(np.float32) # (50, 260) -> (260, 50)
        image = np.expand_dims(image, axis=0) # (260, 50) -> (1, 260, 50)

        return torch.tensor(image, dtype = torch.float), torch.tensor(target, dtype= torch.long)
        