# Watermark (p3)

Adds an IITD logo watermark to the bottom-right of an image with 50% transparency.

Mask and blend:
- Build a binary mask M by thresholding near-white background (M=1 on logo, 0 on white).
- Resize logo width to 20% of document width (mask resized the same).
- Blend at bottom-right using: O = (1 − M)·D + M·0.5·(D + L), where D=document ROI, L=logo.

Usage:
- python watermark.py <input_image> [logo_image] [output_image]
  - Defaults logo to iitlogo-23.jpg in this folder
  - Defaults output to results/watermarked.jpg

Output:
- results/watermarked.jpg
- results/logo_mask.jpg (mask visualization)

Dependencies:
- Python 3, OpenCV (cv2), NumPy
