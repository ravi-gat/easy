from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')
results = model('cat.jpg')

img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis('off')
plt.show()