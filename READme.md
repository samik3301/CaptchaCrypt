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

### How to run the model and other scripts-
Run the `train.py` script for model training.

Run the `cursor_coordinates.py` to get the coordinates position of cursor dynamically and use that to get the top left and bottom right coordinates of the VTOP Captcha box 

Feed those coordinates into the `dataset_maker.py` to generate a dataset of VTOP captchas (Works on Windows, Issues on MacOS for some reason - `pyautogui.screenshot()` not working properly for MacOS)

Run the `load_model.py` to load the saved weights (previous model)

*Possible errors- model not getting saved or getting a pickle TypeError after training* **[FIXED]**

