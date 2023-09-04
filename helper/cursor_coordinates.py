import pyautogui
import time
#no need for importing os-  just position function is enough

for i in range(10):
    print(f"The script is starting in {i+1} seconds.")
    time.sleep(1)

try:
    while True:
        x, y = pyautogui.position()
        print(f"Mouse cursor position: X={x}, Y={y}")
        time.sleep(0.5)  # Adjust the sleep interval (in seconds) as needed
except KeyboardInterrupt:
    print("\nScript terminated by Cntrl+C") #press control+c to stop getting the positions