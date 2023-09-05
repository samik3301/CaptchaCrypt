import os
images = "images"
images_train = "train"

labels = "labels"
labels_train = "train"

parent_dir = "./data"
  
path = os.path.join(parent_dir, images)
os.mkdir(path) #made data/images

path2 = os.path.join(path,images_train)
os.mkdir(path2) #made data/images/train

path3 = os.path.join(parent_dir,labels)
os.mkdir(path3) #made data/label

path4 = os.path.join(path3,labels_train)
os.mkdir(path4) #made data/label/train

#LMAO DED what am i doing smh

