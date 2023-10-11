DATA_DIR = '../data/vtop_captchas/'
MODEL_PATH = '../model_saves/vtop_model.bin'
BATCH_SIZE = 64  #try and change to 16 later and see  
IMAGE_WIDTH = 260
IMAGE_HEIGHT = 50
EPOCHS = 200 #keeping it low for initial training -> will increase later
DEVICE = "cuda:0"