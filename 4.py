import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('scene2.jpg', 0)

# Fourier Transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
mag = 20 * np.log(np.abs(fshift) + 1)

# Low-pass filter
r, c = img.shape
mask = np.zeros((r, c), np.uint8)
mask[r//2-30:r//2+30, c//2-30:c//2+30] = 1

filtered = fshift * mask
result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(mag, cmap='gray')
plt.title('Fourier Transform')

plt.subplot(1, 3, 3)
plt.imshow(result, cmap='gray')
plt.title('Low-Pass Filter')

plt.show()