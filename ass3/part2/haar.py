#!/usr/bin/env python3
import sys
import os
import math
import numpy as np
from PIL import Image


def haar1d_forward(x):
    x = x.astype(np.float64)
    n = x.shape[0]
    res = x.copy()
    temp = np.zeros(n, dtype=np.float64)
    lvl = n
    while lvl > 1:
        half = lvl//2
        for i in range(half):
            a = res[2*i]
            b = res[2*i+1]
            temp[i] = (a+b)/math.sqrt(2.0)
            temp[half+i] = (a-b)/math.sqrt(2.0)
        res[:lvl] = temp[:lvl]
        lvl = half
    return res

def haar1d_inverse(t):
    t = t.astype(np.float64)
    n = t.shape[0]
    res = t.copy()
    temp = np.zeros(n, dtype=np.float64)
    lvl = 1
    while lvl < n:
        half = lvl
        for i in range(half):
            avg = res[i]
            diff = res[half+i]
            temp[2*i] = (avg+diff)/math.sqrt(2.0)
            temp[2*i+1] = (avg-diff)/math.sqrt(2.0)
        res[:2*half] = temp[:2*half]
        lvl = lvl*2
    return res

def haar2d_forward(a):
    a = a.astype(np.float64)
    n,m = a.shape
    res = a.copy()
    
    for i in range(n):
        res[i,:] = haar1d_forward(res[i,:])
    
    for j in range(m):
        res[:,j] = haar1d_forward(res[:,j])
    return res

def haar2d_inverse(t):
    t = t.astype(np.float64)
    n,m = t.shape
    res = t.copy()
    
    for j in range(m):
        res[:,j] = haar1d_inverse(res[:,j])
    
    for i in range(n):
        res[i,:] = haar1d_inverse(res[i,:])
    return res

def keep_top_coeffs(t, k):
    flat = t.flatten()
    idx = np.argsort(np.abs(flat))
    keep = np.zeros_like(flat)
    if k>0:
        keep_idx = idx[-k:]
        keep[keep_idx] = flat[keep_idx]
    return keep.reshape(t.shape)

def save_img(arr, path):
    a = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)

def main():
    if len(sys.argv) < 3:
        print('usage: python3 part2/haar.py <input> <outdir>')
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    img = Image.open(inp).convert('L')
    w,h = img.size
    s = min(w,h)
    
    N = 1
    while N*2 <= s:
        N *= 2
    
    left = (w-N)//2
    top = (h-N)//2
    imgc = img.crop((left, top, left+N, top+N))
    f = np.array(imgc, dtype=np.float64)

    t = haar2d_forward(f)

    fnorm = math.sqrt(np.sum(f*f))
    tnorm = math.sqrt(np.sum(t*t))
    print('N=',N,'||f||=',fnorm,'||t||=',tnorm,'diff=',abs(fnorm-tnorm))

    
    save_img(f, os.path.join(out,'orig.png'))

    
    disp = t + 128.0
    save_img(disp, os.path.join(out,'haar_shifted.png'))

    
    total = N*N
    k1 = total//16
    k2 = total//256

    t1 = keep_top_coeffs(t, k1)
    rec1 = haar2d_inverse(t1)
    save_img(rec1, os.path.join(out,'rec_top_1_16.png'))

    
    small = imgc.resize((N//4, N//4), Image.BILINEAR)
    up = small.resize((N, N), Image.BILINEAR)
    save_img(np.array(up, dtype=np.float64), os.path.join(out,'down_up_1_16.png'))

    t2 = keep_top_coeffs(t, k2)
    rec2 = haar2d_inverse(t2)
    save_img(rec2, os.path.join(out,'rec_top_1_256.png'))

    small2 = imgc.resize((N//16, N//16), Image.BILINEAR)
    up2 = small2.resize((N, N), Image.BILINEAR)
    save_img(np.array(up2, dtype=np.float64), os.path.join(out,'down_up_1_256.png'))

    
    save_img(np.abs(t), os.path.join(out,'haar_abs.png'))

    
    mid = f[N//2,:]
    t1d = haar1d_forward(mid)
    inv1d = haar1d_inverse(t1d)
    e1 = np.max(np.abs(inv1d-mid))
    print('1D mid row recon maxerr=',e1)
    print('1D norms: ||mid||=',math.sqrt(np.sum(mid*mid)),'||t1d||=',math.sqrt(np.sum(t1d*t1d)))

    print('done outputs in',out)

if __name__=='__main__':
    main()
