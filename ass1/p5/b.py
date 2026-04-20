import sys
import time
import os
import cv2
import numpy as np
from a import convolve, gaussian_kernel_1d, gaussian_kernel_2d, laplacian_kernel


def ensure_dir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def convolve_naive_loops(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    H, W = img.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect").astype(np.float32)

    out = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            acc = 0.0
            for i in range(kh):
                for j in range(kw):
                    acc += p[y + i, x + j] * kernel[i, j]
            out[y, x] = acc
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_separable(img_u8: np.ndarray, sigma: float, g1) -> np.ndarray:
    # g1 = gaussian_kernel_1d(sigma)
    h = convolve_naive_loops(img_u8, g1.reshape(1, -1))
    v = convolve_naive_loops(h, g1.reshape(-1, 1))
    return v


def main():
    # Accept explicit path or fall back to download.jpeg
    if len(sys.argv) >= 2 and not sys.argv[1].replace('.', '').isdigit():
        img_path = sys.argv[1]
        sigma_arg_start = 2
    else:
        # try defaults
        here = os.path.dirname(__file__)
        candidates = [os.path.join(here, 'download.jpeg'), os.path.join(here, '..', 'download.jpeg')]
        img_path = None
        for c in candidates:
            if os.path.isfile(c):
                img_path = c
                break
        if img_path is None:
            print("Image path not provided and download.jpeg not found. Usage: python b.py <image> [sigma]")
            return
        sigma_arg_start = 1

    sigma = 2.0
    if len(sys.argv) > sigma_arg_start:
        try:
            sigma = float(sys.argv[sigma_arg_start])
        except ValueError:
            pass
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    ensure_dir(out_dir)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load", img_path)
        return

    # Laplacian
    lap = convolve(img, laplacian_kernel(), c=128, mode='standard', padding='reflect')
    cv2.imwrite(os.path.join(out_dir, 'laplacian.jpg'), lap)

    # Gaussian naive 2D
    G2 = gaussian_kernel_2d(sigma)
    t0 = time.time()
    # g_naive = convolve(img, G2, c=0, mode='standard', padding='reflect')
    g_naive = convolve_naive_loops(img, G2)
    t1 = time.time()

    # Gaussian separable
    g1 = gaussian_kernel_1d(sigma)
    t2 = time.time()
    g_sep = gaussian_separable(img, sigma, g1)
    t3 = time.time()

    cv2.imwrite(os.path.join(out_dir, f'gauss_naive_sigma{sigma}.jpg'), g_naive)
    cv2.imwrite(os.path.join(out_dir, f'gauss_sep_sigma{sigma}.jpg'), g_sep)
    print(f"Image: {img_path}\nNaive 2D: {(t1 - t0)*1000:.2f} ms  |  Separable: {(t3 - t2)*1000:.2f} ms (sigma={sigma})")


if __name__ == "__main__":
    main()
