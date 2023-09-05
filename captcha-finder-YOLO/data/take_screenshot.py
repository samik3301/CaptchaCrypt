import os
import time
import pyautogui

ss_dir = "./test" # Directory to save the testing screenshot image -> model input

#loading time for script - good practice
for i in range(10):
    print(f"The script is starting in {i+1} seconds.")
    time.sleep(1)

img_name = f'screenshot_test.png'
screenshot = pyautogui.screenshot()
screenshot.save(os.path.join(ss_dir, img_name))

print(f'Screenshot saved to: {os.path.join(ss_dir, img_name)}')

print("Script terminated successfully.")