#!/usr/bin/env python3
import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def erode_gray(img, se):
    h, w = img.shape
    sh, sw = se.shape
    cy, cx = sh // 2, sw // 2
    
    out = np.zeros_like(img)
    
    for y in range(h):
        for x in range(w):
            vals = []
            for sy in range(sh):
                for sx in range(sw):
                    if se[sy, sx] == 0:
                        continue
                    
                    iy = y + sy - cy
                    ix = x + sx - cx
                    
                    if 0 <= iy < h and 0 <= ix < w:
                        vals.append(img[iy, ix])
            
            if vals:
                out[y, x] = min(vals)
    
    return out

def dilate_gray(img, se):
    h, w = img.shape
    sh, sw = se.shape
    cy, cx = sh // 2, sw // 2
    
    out = np.zeros_like(img)
    
    for y in range(h):
        for x in range(w):
            vals = []
            for sy in range(sh):
                for sx in range(sw):
                    if se[sy, sx] == 0:
                        continue
                    
                    iy = y + sy - cy
                    ix = x + sx - cx
                    
                    if 0 <= iy < h and 0 <= ix < w:
                        vals.append(img[iy, ix])
            
            if vals:
                out[y, x] = max(vals)
    
    return out

def opening_gray(img, se):
    temp = erode_gray(img, se)
    return dilate_gray(temp, se)

def closing_gray(img, se):
    temp = dilate_gray(img, se)
    return erode_gray(temp, se)

def disk(r):
    size = 2 * r + 1
    c = r
    y, x = np.ogrid[:size, :size]
    return ((x - c)**2 + (y - c)**2 <= r**2).astype(np.uint8)

def bottom_hat(img, se):
    c = closing_gray(img, se)
    return c - img

def stretch(img):
    mn, mx = img.min(), img.max()
    if mx > mn:
        return ((img - mn) * 255.0 / (mx - mn)).astype(np.uint8)
    return img

def save(arr, path):
    Image.fromarray(arr).save(path)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <fundus_image> <output_dir>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    out = sys.argv[2]
    
    os.makedirs(out, exist_ok=True)
    
    img = Image.open(img_path)
    if img.mode != 'L':
        arr = np.array(img)
        if len(arr.shape) == 3:
            import cv2
            f = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            f = arr
    else:
        f = np.array(img)
    
    save(f, os.path.join(out, '2_original.png'))
    
    r = 5
    se = disk(r)
    
    f_eroded = erode_gray(f, se)
    f_dilated = dilate_gray(f, se)
    f_opened = opening_gray(f, se)
    f_closed = closing_gray(f, se)
    
    save(se * 255, os.path.join(out, '2a_disk_se.png'))
    save(f_eroded, os.path.join(out, '2a_eroded.png'))
    save(f_dilated, os.path.join(out, '2a_dilated.png'))
    save(f_opened, os.path.join(out, '2a_opened.png'))
    save(f_closed, os.path.join(out, '2a_closed.png'))
    
    r_bh = 12
    se_bh = disk(r_bh)
    vessels = bottom_hat(f, se_bh)
    vessels_stretched = stretch(vessels)
    
    save(vessels, os.path.join(out, '2b_bottom_hat_raw.png'))
    save(vessels_stretched, os.path.join(out, '2b_vessels.png'))
    
    r_gap = 2
    se_gap = disk(r_gap)
    vessels_filled = closing_gray(vessels_stretched, se_gap)
    
    save(vessels_filled, os.path.join(out, '2c_vessels_filled.png'))
    
    sizes = range(1, 16)
    areas = []
    
    v_bin = (vessels_filled > 30).astype(np.uint8)
    
    for r in sizes:
        se_open = disk(r)
        opened = opening_gray(v_bin * 255, se_open)
        area = np.sum(opened > 0)
        areas.append(area)
    
    diffs = []
    for i in range(len(areas) - 1):
        diffs.append(areas[i] - areas[i+1])
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sizes, areas, 'b-o')
    plt.xlabel('SE Radius (pixels)')
    plt.ylabel('Area After Opening')
    plt.title('Granulometry: Area vs SE Size')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(sizes[:-1], diffs, 'r-o')
    plt.xlabel('SE Radius (pixels)')
    plt.ylabel('Area Removed')
    plt.title('Pattern Spectrum: Vessel Thickness Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out, '2d_granulometry.png'), dpi=150)
    plt.close()
    
    print(f"Saved results to {out}/")

if __name__ == "__main__":
    main()
