import os
from retrieve_captcha import get_captcha
import time
import random

NUM_IMAGES = 200

URL = "https://www.india.gov.in/user/login"
data_dir = "./../data/indiagovin_captchas"

itr = 1
while itr <= NUM_IMAGES:
    try:
        captcha = get_captcha(URL)
        img_name = f'save_{itr}.png'
        captcha.save(os.path.join(data_dir, img_name))
        time.sleep(random.randint(4, 8))
        itr += 1
    except KeyboardInterrupt:
        print("Script terminated by interupt forcefully")

print(f"Saved {itr - 1} captchas")
