from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # loading the nano model

# Use the model
results = model.train(data="config.yaml", epochs=1)  # train the model
#train for more number of epochs to get proper training results in graphs over epochs
#how to know if the model is working right- > the loss graphs will decrease over the number of epochs
#check the results.png in /runs/detect/train after increasing the number of epochs to 100 and training over a larger dataset