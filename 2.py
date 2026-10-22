import cv2
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpg", 0)

eq = cv2.equalizeHist(img)

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(eq, cmap='gray')
plt.title('Equalized')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.hist(img.ravel(), 256, [0, 256])
plt.title('Original Histogram')

plt.subplot(2, 2, 4)
plt.hist(eq.ravel(), 256, [0, 256])
plt.title('Equalized Histogram')

plt.tight_layout()
plt.show()