DATA_DIR = '../data/captcha_images_v2/'
MODEL_PATH = '../model_saves/model.bin'
BATCH_SIZE = 32  #try and change to 16 later and see  
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
# NUM_WORKERS = 2 #depends on the device
EPOCHS = 200 #keeping it low for initial training -> will increase later
DEVICE = "cuda:0"