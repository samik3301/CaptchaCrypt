import glob
filenames = glob.glob("./data/indiagov_captchas_ultimate/*.png") #Changed the dataset path to the ultimate ~1100 captcha
count=0
for f in filenames:
    img_name = f.split("/")[-1][:-4]  #changed \\ - > / for Mac
    if len(img_name) != 6:
        print(f"Improper annotation labeled : {img_name}")
        count+=1
if count==0:
    print("Good work, no improper annotations in the dataset!")
exit(0)