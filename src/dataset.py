import albumentations
import torch
import numpy as np


from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Classification:
    def __init__(self,image_paths, targets, resize=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        # resizing gonna give a tuple of (h,w)
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(always_apply=True)
            ]
        )

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,item):
        image = Image.open(self.image_paths[item]).convert("RGB") #changing the 4 channel input image into rgb 
        targets = self.targets[item] #setting the targets
        if self.resize!=None: #in case its not resized -> this will resize 
            image= image.resize((self.resize[1],self.resize[0]),resample=Image.BILINEAR)
            #since PIL format saves the image as height first then width so resizing accordingly 
            #and also using resampling method of PIL as BILINEAR -> can change that later to see performance improvements if any
        image = np.array(image) #converting the image into np array finally  
        augmented = self.aug(image= image)
        image = augmented["image"] #image becomes the augmented image

        image = np.transpose(image, (2,0,1)).astype(np.float32)

        return{
            "images" : torch.tensor(image, dtype = torch.float),
            "targets": torch.tensor(targets,dtype= torch.long)
        }  #returning both the image tensor and the target tensors

    