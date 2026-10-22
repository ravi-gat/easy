import torch
import torchvision
import cv2
import matplotlib.pyplot as plt

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)

model.eval()

img = cv2.imread("cat.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

tensor = torchvision.transforms.functional.to_tensor(img)

with torch.no_grad():
    output = model([tensor])[0]

for box, score in zip(output["boxes"], output["scores"]):
    if score > 0.5:
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.imshow(img)
plt.axis("off")
plt.show()