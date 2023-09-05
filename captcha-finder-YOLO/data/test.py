from ultralytics import YOLO
import os
import cv2


model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

img_dir = "./test"
#subdir=os.listdir(img_dir)
for file in img_dir:
    frame = cv2.imread(os.path.join(img_dir, file)) 

model = YOLO(model_path)  # loading our custom saved model 

threshold = 0.5 #threshold for successful prediction -> fine tune this if needed

results = model(frame)

#results.names[int(class_id)].upper()
class_name = "captcha"

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4) #drawing a bounding box
        cv2.putText(frame, class_name , (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        

print(f"The top left coordinates of the bounding box: ({x1},{y1})")
print(f"The bottom right coordinates of the bounding box: ({x2},{y2})")


