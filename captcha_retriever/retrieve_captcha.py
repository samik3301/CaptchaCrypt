import urllib.request
from PIL import Image

def get_secureimage_link(res_text: str) -> str:
    start = res_text.find("securimage?sid=")
    if start == -1:
        print("Secureimage string not found !!")
        exit(0)
    
    buf = ""
    while res_text[start] != "\"":
        buf += res_text[start]
        start = start + 1

    link = "https://www.india.gov.in/" + buf
    return link

def get_captcha(url :str) -> Image.Image:
    # URL = "https://www.india.gov.in/user/login"
    response = urllib.request.urlopen(url)
    res_text = str(response.read())

    img_link = get_secureimage_link(res_text)
    img = Image.open(urllib.request.urlopen(img_link))
    return img

