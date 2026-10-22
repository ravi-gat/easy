import cv2
import matplotlib.pyplot as plt

img = cv2.imread('cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edge = cv2.Laplacian(gray, cv2.CV_64F)
edge = cv2.convertScaleAbs(edge)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge, cmap='gray')
plt.title('Laplacian Edges')
plt.axis('off')

plt.show()