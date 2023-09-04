import os
import time
import pyautogui
import random
#from PIL import ImageGrab

#the chatGPT stuff didnt work so had to make this from scratch ded
# https://pyautogui.readthedocs.io/en/latest/screenshot.html [documentation that i refered]

#https://stackoverflow.com/questions/76361049/how-to-fix-typeerror-not-supported-between-instances-of-str-and-int-wh



#bounding box of vtop captcha

#change the coordinates for ur device resolution using the cursor_coordinates.py 
x1, y1 = 612, 372  # Top-left corner 
x2, y2 = 803, 407  # Bottom-right corner 

#niga do ur magic and turn this into a relative path
download_dir = "/Users/samik/Desktop/Programming/CaptchaCrypt/downloaded_images" # Directory to save downloaded images
#os.makedirs(download_dir, exist_ok=True)

#loading time for script - good practice
for i in range(10):
    print(f"The script is starting in {i+1} seconds.")
    time.sleep(1)


x = 2 #can specify how many images we need for our dataset
i = 0 

while x>0:
    try:
        #timestamp = time.strftime('%Y-%m-%d_%H-%M-%S') #hopefully should work
        img_name = f'screenshot_{i+1}.png'
        i+=1
        screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1)) #top left, width,height for the rectangle box
        # Capture the image from the specified coordinates
        screenshot.save(os.path.join(download_dir, img_name))
        # Reload the webpage 
        pyautogui.hotkey("command", "r")  # Refresh the page (coded for mac-> change command with ctrl for windows)
        time.sleep(5)  # Wait for the page to load - 5s is enough ig unless really bad internet

        print(f'Screenshot saved to: {os.path.join(download_dir, img_name)}')

        # Wait for some time before reloading again
        time.sleep(random.randint(5,10))  # keeping it random so vtop doesn't flag us or something
        x-=1

    except KeyboardInterrupt:
      break

print("Script terminated forcefully.")
