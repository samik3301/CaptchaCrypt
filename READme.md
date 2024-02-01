## CaptchaCrypt - Testing Strengths of Captchas
### With help of Deep Learning

This implementation makes use of RCNN architecture with CTC loss function to predict and digitally retreive the Captchas.

*Make a virtual environment for Python followed by activating that environment.*

For Linux and MacOS-
*Run the following command.*

`pip install -r requirements_keras.txt`

### How to get the Captcha from any website [Updated]

To find this, navigate into the /captcha_retreiver director and run the `create_dataset_captcha.py` script. The website used can be changed accordingly to how vulnerable a website is regarding their CAPTCHA abstraction.

In our implementation we retreived and made our collected dataset of CAPTCHAs for VTOP (https://vtop.ac.in). Followed by manual annotation (labeling the name of each image same as their ground truth captcha) [very tedious and hard process :( ]

### How to run the project : 

Run the `keras_implementation.ipynb` script, cell by cell. Number of iterations to be trained for can be changed accordingly to user's need. 

### Checkpoint-
Can integrate this with an interface in future or make it into an extension that can be used like a Google Chrome extension.

### Detailed Report - 
For an in depth report about the project, kindly refer the link below.
[Report][https://docs.google.com/document/d/1uQYvcg5S8hhhf1-egFX8zbuXI53lKWaFslWqsLyvsY4/edit?usp=sharing]