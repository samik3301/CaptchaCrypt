import requests
import urllib.request
import base64

def string_between(text: str, a: str, b: str):
    sec = text.split(a)[1]
    print(sec.split(b)[0])
    return text.split(a)[1].split(b)[0]

def base64_to_img(b64: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.urlsafe_b64decode(b64))

URL = "https://vtopcc.vit.ac.in/vtop/login"
response = urllib.request.urlopen(URL)
res_text = str(response.read())
with open("./temp.txt", "w") as f:
    f.write(res_text)
#print(res_text)
#base64_img = string_between(res_text, ";base64,", "\"")
#print(f"Image : ", {base64_img})
