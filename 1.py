import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('cat.jpg')

# Gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# CMYK
rgb = img / 255.0
C = 1 - rgb[:, :, 2]
M = 1 - rgb[:, :, 1]
Y = 1 - rgb[:, :, 0]
K = np.minimum(np.minimum(C, M), Y)

# Display images
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Gray')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
plt.title('HSV')

plt.subplot(1, 3, 3)
plt.imshow(K, cmap='gray')
plt.title('CMYK - Black')

plt.show()