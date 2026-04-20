#!/usr/bin/env python3
import sys
import os
import numpy as np
from PIL import Image

def dilate(a, se):
    img = a.astype(bool)
    se = se.astype(bool)
    
    h, w = se.shape
    cy, cx = h // 2, w // 2
    
    out = np.zeros_like(img, dtype=bool)
    coords = np.argwhere(se) - [cy, cx]
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j]:
                for dy, dx in coords:
                    ni, nj = i + dy, j + dx
                    if 0 <= ni < out.shape[0] and 0 <= nj < out.shape[1]:
                        out[ni, nj] = True
    
    return out.astype(np.uint8)

def erode(a, se):
    img = a.astype(bool)
    se = se.astype(bool)
    
    h, w = se.shape
    cy, cx = h // 2, w // 2
    
    pad = np.pad(img, ((cy, cy), (cx, cx)), mode='constant', constant_values=0)
    out = np.zeros_like(img, dtype=bool)
    coords = np.argwhere(se)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            match = True
            for sy, sx in coords:
                pi, pj = i + sy, j + sx
                if not pad[pi, pj]:
                    match = False
                    break
            out[i, j] = match
    
    return out.astype(np.uint8)

def opening(a, se):
    temp = erode(a, se)
    return dilate(temp, se)

def closing(a, se):
    temp = dilate(a, se)
    return erode(temp, se)

def hit_or_miss(a, se1, se2):
    h1 = erode(a, se1)
    h2 = erode(1 - a, se2)
    return (h1 & h2).astype(np.uint8)

def disk(r):
    size = 2 * r + 1
    c = r
    y, x = np.ogrid[:size, :size]
    return ((x - c)**2 + (y - c)**2 <= r**2).astype(np.uint8)

def make_rgb(a, detected):
    # a and detected are 0/1 arrays; return 0/1 RGB overlay
    h, w = a.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = a
    rgb[:, :, 1] = detected
    rgb[:, :, 2] = detected
    return rgb

def save(arr, path):
    img = Image.fromarray(arr * 255)
    img.save(path)

def main():
    if len(sys.argv) != 4:
        print(f"Usage: python3 {sys.argv[0]} <binary_image> <c_image> <output_dir>")
        sys.exit(1)
    
    bin_path = sys.argv[1]
    c_path = sys.argv[2]
    out = sys.argv[3]
    
    os.makedirs(out, exist_ok=True)
    
    bin_img = np.array(Image.open(bin_path).convert('L'))
    c_img = np.array(Image.open(c_path).convert('L'))
    
    a = (bin_img == 255).astype(np.uint8)
    b = (c_img == 255).astype(np.uint8)
    
    r = 2
    se_disk = disk(r)
    
    a_dilated = dilate(a, se_disk)
    a_closed = closing(a, se_disk)
    
    save(a, os.path.join(out, '1a_original.png'))
    save(se_disk, os.path.join(out, '1a_disk_se.png'))
    save(a_dilated, os.path.join(out, '1a_dilated.png'))
    save(a_closed, os.path.join(out, '1a_closed.png'))
    
    detected_b = opening(a, b)
    save(detected_b, os.path.join(out, '1b_opening.png'))
    rgb_b = make_rgb(a, detected_b)
    save(rgb_b, os.path.join(out, '1b_opening_rgb.png'))
    
    small = disk(1)
    b1 = erode(b, small)
    b1 = erode(b1, small)
    
    if np.sum(b1) < 10:
        b1 = erode(b, small)
    
    detected_c = opening(a, b1)
    save(b, os.path.join(out, '1c_b_original.png'))
    save(b1, os.path.join(out, '1c_b1_eroded.png'))
    save(detected_c, os.path.join(out, '1c_opening_b1.png'))
    rgb_c = make_rgb(a, detected_c)
    save(rgb_c, os.path.join(out, '1c_opening_b1_rgb.png'))
    
    b2 = np.zeros_like(b)
    h, w = b.shape
    
    mid_h_start = h // 2 - 3
    mid_h_end = h // 2 + 3
    right_start = int(w * 0.5)
    right_end = int(w * 0.85)
    
    b2[mid_h_start:mid_h_end, right_start:right_end] = 1
    b2 = dilate(b2, disk(1))
    
    hmt = hit_or_miss(a, b1, b2)
    recon = dilate(hmt, b)
    
    save(b2, os.path.join(out, '1d_b2_miss.png'))
    save(hmt, os.path.join(out, '1d_hmt_points.png'))
    save(recon, os.path.join(out, '1d_detected.png'))
    rgb_d = make_rgb(a, recon)
    save(rgb_d, os.path.join(out, '1d_detected_rgb.png'))
    
    print(f"Saved results to {out}/")

if __name__ == "__main__":
    main()