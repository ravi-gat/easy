import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpg", 0)

edges = cv2.Canny(img, 50, 150)
lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)

        x = int(a * rho)
        y = int(b * rho)

        cv2.line(img, (x-500, y-500),
                 (x+500, y+500), (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()