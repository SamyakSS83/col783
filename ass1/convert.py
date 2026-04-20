import sys
import cv2
import numpy as np

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <image_path>")
    sys.exit(1)

img = cv2.imread(sys.argv[1])              # Read image (BGR)
if img is None:
    print("Error: Could not read image.")
    sys.exit(1)

bw = np.mean(img[:, :, :3], axis=2).astype(np.uint8)  # Average channels
cv2.imwrite("bw_image.png", bw)            # Save result
