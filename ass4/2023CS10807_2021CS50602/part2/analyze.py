#!/usr/bin/env python3
import numpy as np
from PIL import Image
import cv2

img = Image.open('f.png')
arr = np.array(img)

print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"Min: {arr.min()}, Max: {arr.max()}")
print(f"Mean: {arr.mean():.1f}")

if len(arr.shape) == 3:
    print(f"Channels: {arr.shape[2]}")
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
else:
    gray = arr

print(f"\nGrayscale stats:")
print(f"Min: {gray.min()}, Max: {gray.max()}")
print(f"Mean: {gray.mean():.1f}, Std: {gray.std():.1f}")

hist, bins = np.histogram(gray, bins=50, range=(0, 256))
print(f"\nHistogram peaks:")
for i in np.argsort(hist)[-5:]:
    print(f"  Bin {bins[i]:.0f}-{bins[i+1]:.0f}: {hist[i]} pixels")

g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
Image.fromarray(g).save('gray.png')
print(f"\nSaved grayscale version to gray.png")
