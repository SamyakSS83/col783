import sys
import os
import numpy as np
import cv2


def pad(img: np.ndarray, ph: int, pw: int, mode: str = "reflect") -> np.ndarray:
    if mode == "reflect":
        return np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    if mode == "edge":
        return np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    return np.pad(img, ((ph, ph), (pw, pw)), mode="constant")  # zeros


def convolve(img_u8: np.ndarray,
             kernel: np.ndarray,
             c: int = 0,
             mode: str = "standard",
             padding: str = "reflect") -> np.ndarray:
    if img_u8.dtype != np.uint8:
        raise ValueError("img must be uint8")
    k = kernel.astype(np.float32)
    k = k[::-1, ::-1]
    kh, kw = k.shape
    H, W = img_u8.shape
    img = img_u8.astype(np.float32)

    if mode == "standard":
        ph, pw = kh // 2, kw // 2
        p = pad(img, ph, pw, padding)
        out = np.empty((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                region = p[y:y+kh, x:x+kw]
                out[y, x] = float(np.sum(region * k))
    elif mode == "full":
        out_h, out_w = H + kh - 1, W + kw - 1
        p = np.pad(img, ((kh-1, kh-1), (kw-1, kw-1)), mode="constant")
        out = np.zeros((out_h, out_w), dtype=np.float32)
        for y in range(out_h):
            for x in range(out_w):
                region = p[y:y+kh, x:x+kw]
                out[y, x] = float(np.sum(region * k))
    else:
        raise ValueError("mode must be standard or full")

    out = out + float(c)
    return np.clip(out, 0, 255).astype(np.uint8)


def convolve_float(img: np.ndarray, kernel: np.ndarray, padding: str = "reflect") -> np.ndarray:
    # Same as standard convolution but returns float32 (no clipping / constant).
    k = kernel.astype(np.float32)[::-1, ::-1]
    kh, kw = k.shape
    H, W = img.shape
    ph, pw = kh // 2, kw // 2
    p = pad(img.astype(np.float32), ph, pw, padding)
    out = np.empty((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            out[y, x] = float(np.sum(p[y:y+kh, x:x+kw] * k))
    return out


def laplacian_kernel():
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-(x * x) / (2 * sigma * sigma))
    g /= g.sum()
    return g


def gaussian_kernel_2d(sigma: float) -> np.ndarray:
    g1 = gaussian_kernel_1d(sigma)
    g2 = np.outer(g1, g1)
    g2 /= g2.sum()
    return g2.astype(np.float32)


def _demo():
    img = np.arange(25, dtype=np.uint8).reshape(5, 5) * 5
    mean3 = (1/9.0) * np.ones((3, 3), dtype=np.float32)
    print("Input:\n", img)
    print("Mean3 conv:\n", convolve(img, mean3))
    print("Laplacian + 128:\n", convolve(img, laplacian_kernel(), c=128))
    print("Full conv (laplacian) shape:", convolve(img[:2, :2], laplacian_kernel(), mode='full').shape)

    # Image demo using download.jpeg if available
    here = os.path.dirname(__file__)
    candidates = [os.path.join(here, 'download.jpeg'), os.path.join(here, '..', 'download.jpeg')]
    chosen = None
    for c in candidates:
        if os.path.isfile(c):
            chosen = c
            break
    if chosen:
        gray = cv2.imread(chosen, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            out_dir = os.path.join(here, 'results')
            os.makedirs(out_dir, exist_ok=True)
            mean_blur = convolve(gray, mean3)
            lap = convolve(gray, laplacian_kernel(), c=128)
            cv2.imwrite(os.path.join(out_dir, 'download_mean.jpg'), mean_blur)
            cv2.imwrite(os.path.join(out_dir, 'download_lap.jpg'), lap)
            print(f"Processed '{chosen}' -> results/download_mean.jpg & download_lap.jpg")
        else:
            print(f"Found '{chosen}' but could not load as image")
    else:
        print("No download.jpeg found (skipping image demo)")


if __name__ == "__main__":
    _demo()
