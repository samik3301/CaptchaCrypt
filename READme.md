## CaptchaCrypt - Testing Strengths of Captchas
### With help of Deep Learning

This PyTorch implementation makes use of CRNN architecture with CTC loss function to predict and read the Captchas.

*First things first-Make a virtual environment for **Python 3.8.16** & activate that environment and do everything within that.*

**Edit the DATA_DIR (make it relative path), DEVICE (to cuda or whatever your system has) and WORKERS (again according to your system's specification)(not needed really)**


*Run this command in the terminal to download the dataset within the same repository directory.*
For Linux and MacOS-

`curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip`
`unzip -qq captcha_images_v2.zip`

*Run the following command.*

`pip install -r requirements.txt`

Changes and progress pending for upcoming commits- 
- Training the model with 200 Epochs minimum **[Done]**
- Making custom **vtopcaptcha** dataset **[Done]**
- Custom vtop_captchas dataset made, now need to manually annotate the labels as filename
- Train the model onto vtop_captchas 
- Fixing if any model save issues, subjected to successful training over the required epochs **[Done]**
- Need to save the new model (VTOP Captcha one with a different model name [DON'T UPDATE THE PREVIOUS])
- Fixed the model saving as .bin (weights), need to save it as a `.pt` or `.pth` file after loading the model(try idk)
- Making changes to `decode_predictions()` function in the model to make our output more presentable
- Optimize using the mentioned comments within code and testing stuff - hyperparameter tuning, LSTM instead of GRU, learning rate, Optimizers, Batch Size, Convolutional Layers etc
- Debugging if any errors **[No Errors so far]**
- Change the image height and wdith to fix the feature length after permute from 75 -> 20 
- Automating the process of finding captcha on any website and feeding that as testing input to our model
- Deployment via FLASK web server or as an Extension 
- Need to make scripts callable by one main script (import other scripts into one)

### How to run the model and other scripts-
Run the `train.py` script for model training.

Run the `cursor_coordinates.py` to get the coordinates position of cursor dynamically and use that to get the top left and bottom right coordinates of the VTOP Captcha box 

Feed those coordinates into the `dataset_maker.py` to generate a dataset of VTOP captchas (Works on Windows, Issues on MacOS for some reason - `pyautogui.screenshot()` not working properly for MacOS)

Run the `load_model.py` to load the saved weights (previous model)

### How to get the Captcha box coordinates (bounding box) from any website [Updated]
We use a YOLOv8 Object detection model to train on a custom dataset of different websites having captchas and manually and precisely annotating the captchas in those. Then for testing, we take a screenshot of any website -> pass that screenshot as input to our trained model and it should label and detect the captcha present in it :D

We get the coordinates from that bounding box using OpenCV help and feed those into our script to download our testing captcha image that we feed in our CRNN model to predict the label :D

Run the script `dir_maker.py` to make the needed folders that should contain the training data [screenshot images of different websites containing captchas] and their annotations [to be moved later].

[Annotation Tool - cvat.ai](https://www.cvat.ai/)

Make an account and create a new project "CaptchaDetector" or whatever - then a new task-name it captcha-detection-v001 

Then add the images from the `/images/train` , images into the select files on cvat

Once loaded the images - submit and open, Name our label as crack, then click on the *Job #123456* and start the annotations.
Annotations are done using  Rectangle->Shape to make a bounding box around the captcha you see in the image. So be *very careful about what you annotate as a captcha by making a bounding box as it is crucial for our model(it learns from this)* Need v precise, manual labour ded but ok.

Note: Keep saving after every image to make sure no data is lost and is synced to the cloud server. Sadly it doesn't have an autosave so you need to do it manually.

Once this is done for all the images, click tasks -> under the actions -> Export the task dataset -> Select Export format as YOLO 1.1

Move the screenshot images that were collected to be train the YOLO model  into the `/data/images/train` and our annotations will be inside a folder [after unzipping the downloaded dataset] called `obj_train_data` and now move them inside `/data/labels/train`.

**Note that the annotations will have a .txt extension and the name will be same as their respective image.**

Just to be sure check the size of `/data/images/train` and `/data/labels/train`. They should be equal obviously.

After the annotations are done : 
Run `main.py` 

Once the script is ran , the model begins training in the terminal locally using system's resources and the results of the model training are then saved onto `/data/runs/detect/train/`. 


There **might** be a chance that the YOLO part model training gives an error as I am unsure if it saved it into `/data/runs/detect/train/` or `/code/data/runs/detect/train/` so ded.

### Checkpoint-
If it runs properly and saves the model results then proceed with running `take_screenshot.py` which saves our testing data as a screenshot.

After that is done - run `test.py` to test the model with our testing input and it should finally return our the bounding box coordinates needed.
