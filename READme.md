## CaptchaCrypt - Testing Strengths of Captchas
### With help of Deep Learning

This PyTorch implementation makes use of R-CNN architecture with CTC loss function to predict and read the Captchas.

**Edit the DATA_DIR (make it relative path), DEVICE (to cuda or whatever your system has) and WORKERS (again according to your system's specification)**


*Run this command in the terminal to download the dataset within the same repository directory.*

`curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip`

`unzip -qq captcha_images_v2.zip`

*Run the following command.*

`pip install -r requirements.txt`

Changes and progress pending for upcoming commits- 
- Training the model with 200 Epochs minimum
- Making custom **vtopcaptcha** dataset and training the model onto that
- Fixing if any model save issues, subjected to successful training over the required epochs
- Making changes to `decode_predictions()` function in the model to make our output more presentable
- Optimize using the mentioned comments within code and testing stuff - hyperparameter tuning, LSTM instead of GRU, learning rate, Optimizers, Batch Size, Convolutional Layers etc
- Debugging if any errors


### How to run the model-
Run the `train.py` script for model training.

*Possible errors- model not getting saved or getting a pickle TypeError after training*

