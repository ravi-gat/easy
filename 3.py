import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpg', 0)
kernel = np.ones((3, 3), np.uint8)

plt.subplot(2, 2, 1)
plt.imshow(cv2.erode(image, kernel), cmap='gray')
plt.title('Erosion')

plt.subplot(2, 2, 2)
plt.imshow(cv2.dilate(image, kernel), cmap='gray')
plt.title('Dilation')

plt.subplot(2, 2, 3)
plt.imshow(cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel), cmap='gray')
plt.title('Opening')

plt.subplot(2, 2, 4)
plt.imshow(cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel), cmap='gray')
plt.title('Closing')

plt.tight_layout()
plt.show()