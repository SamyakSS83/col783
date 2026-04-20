import os
import cv2
import numpy as np
from typing import Tuple

# Tunables (user can modify these globals):
gamma = 0.25 
r = 7        


def ensure_dir(p: str) -> None:
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def load_hdr_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to load {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def make_disk_kernel(radius: int) -> np.ndarray:
    # Return normalized disk kernel w(x,y) = c if x^2 + y^2 < r^2 else 0, with sum(w)=1.
    if radius < 1:
        return np.array([[1.0]], dtype=np.float32)
    d = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = (x * x + y * y) <= (radius * radius)
    k = np.zeros((d, d), dtype=np.float32)
    k[mask] = 1.0
    s = k.sum()
    if s > 0:
        k /= s
    return k


def convolve_same(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    # Same-size convolution with reflect padding.
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = cv2.copyMakeBorder(img, ph, ph, pw, pw, borderType=cv2.BORDER_REFLECT)
    out = cv2.filter2D(pad, -1, k[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)
    return out[ph:ph+img.shape[0], pw:pw+img.shape[1]]


def to_display_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def gamma_map(x: np.ndarray, g: float) -> np.ndarray:
    eps = 1e-8
    xn = x.astype(np.float32)
    # Rescale HDR to [0,1] by robust percentile to avoid outliers dominating
    lo, hi = np.percentile(xn, 1.0), np.percentile(xn, 99.0)
    if hi <= lo:
        lo, hi = float(xn.min()), float(xn.max())
    x01 = np.clip((xn - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    y = np.power(x01 + eps, g)
    return y


def main():
    here = os.path.dirname(__file__)
    in_path = os.path.join(here, 'nave.hdr')
    out_dir = os.path.join(here, 'results')
    ensure_dir(out_dir)

    hdr = load_hdr_gray(in_path)
    k = make_disk_kernel(int(r))

    # 6a demonstration (linear case): formula T'(x) = a x + b S
    a, b = 1.0, 30.0
    S = float(k.sum())
    # Compute w * T(f)
    wf = convolve_same(hdr, k)
    wTf = convolve_same(a * hdr + b, k)
    # Compute T'(w*f)
    Tp_wf = a * wf + b * S
    # Save a quick diagnostic montage scaled to 8-bit
    cv2.imwrite(os.path.join(out_dir, 'p6_6a_wTf.jpg'), to_display_u8(wTf))
    cv2.imwrite(os.path.join(out_dir, 'p6_6a_Tp_wf.jpg'), to_display_u8(Tp_wf))
    print("Difference between T'(w*f) and w*T(f) (max, mean):", np.max(np.abs(Tp_wf - wTf)), np.mean(np.abs(Tp_wf - wTf)))

    # 6b: nonlinear gamma before/after blur
    g_only = gamma_map(hdr, gamma)
    # gamma then blur
    g_then_blur = convolve_same(g_only, k)
    # blur then gamma
    blur_then_g = gamma_map(convolve_same(hdr, k), gamma)

    # Map to [0,255] for saving
    cv2.imwrite(os.path.join(out_dir, 'p6_gamma.jpg'), to_display_u8(255.0 * g_only))
    cv2.imwrite(os.path.join(out_dir, 'p6_gamma_then_blur.jpg'), to_display_u8(255.0 * g_then_blur))
    cv2.imwrite(os.path.join(out_dir, 'p6_blur_then_gamma.jpg'), to_display_u8(255.0 * blur_then_g))

    print('Saved Part 6 outputs to', out_dir)



if __name__ == '__main__':
    main()
