import sys
import os
import numpy as np
import cv2
from a import convolve_float, gaussian_kernel_1d, gaussian_kernel_2d, laplacian_kernel, convolve


def ensure_dir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def kernel_convolve(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    # Full float convolution of two small kernels (k1 * k2).
    h1, w1 = k1.shape
    h2, w2 = k2.shape
    out = np.zeros((h1 + h2 - 1, w1 + w2 - 1), dtype=np.float32)
    f1 = k1[::-1, ::-1]
    # f2 = k2[::-1, ::-1]
    f2 = np.flip(k2)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            # overlap region
            acc = 0.0
            for i in range(h1):
                yy = y - i
                if 0 <= yy < h2:
                    for j in range(w1):
                        xx = x - j
                        if 0 <= xx < w2:
                            acc += f1[i, j] * f2[yy, xx]
            out[y, x] = acc
    return out


def center_crop_to(shape, arr):
    H, W = shape
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    top = cy - H // 2
    left = cx - W // 2
    return arr[top:top+H, left:left+W]


def montage_with_labels(images, titles, gap=30, top_margin=30, font_scale=0.6):
    assert len(images) == len(titles)
    h = max(im.shape[0] for im in images)
    w_list = [im.shape[1] for im in images]
    total_w = sum(w_list) + gap * (len(images) - 1)
    canvas = np.full((h + top_margin, total_w), 255, dtype=np.uint8)
    x = 0
    for im, title, w in zip(images, titles, w_list):
        # center vertically
        y_offset = top_margin + (h - im.shape[0]) // 2
        canvas[y_offset:y_offset+im.shape[0], x:x+im.shape[1]] = im
        # put title centered over this region
        txt_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        tx = x + (w - txt_size[0]) // 2
        ty = (top_margin + txt_size[1]) // 2  # vertically centered in margin
        cv2.putText(canvas, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, 1, cv2.LINE_AA)
        x += w + gap
    return canvas


def log_variant_a(img_u8: np.ndarray, sigma: float, add: float = 128.0) -> np.ndarray:
    """Variant A: (L * G) * f  (kernel convolution first, then apply once)."""
    L = laplacian_kernel()
    G = gaussian_kernel_2d(sigma)
    LG_full = kernel_convolve(L, G)
    # LG = center_crop_to(G.shape, LG_full)
    f_float = convolve_float(img_u8.astype(np.float32), LG_full, padding='reflect')
    f_float += add
    return np.clip(f_float, 0, 255).astype(np.uint8)


def log_variant_b(img_u8: np.ndarray, sigma: float, add: float = 128.0) -> np.ndarray:
    """Variant B: L * (G * f)."""
    L = laplacian_kernel()
    G = gaussian_kernel_2d(sigma)
    f_float = img_u8.astype(np.float32)
    Gf = convolve_float(f_float, G, padding='reflect')
    out = convolve_float(Gf, L, padding='reflect')
    out = np.clip(out, 0, 255).astype(np.uint8)
    out += int(add)
    return out



def log_variant_c(img_u8: np.ndarray, sigma: float, add: float = 128.0) -> np.ndarray:
    """Variant C: G * (L * f)."""
    L = laplacian_kernel()
    G = gaussian_kernel_2d(sigma)
    f_float = img_u8.astype(np.float32)
    Lf = convolve_float(f_float, L, padding='reflect')
    out = convolve_float(Lf, G, padding='reflect')
    out = np.clip(out, 0, 255).astype(np.uint8)
    out += int(add)
    return out


def main():
    if len(sys.argv) >= 2 and not sys.argv[1].replace('.', '').isdigit():
        img_path = sys.argv[1]
        sigma_pos = 2
    else:
        here = os.path.dirname(__file__)
        candidates = [os.path.join(here, 'image.jpeg'), os.path.join(here, '..', 'image.jpeg')]
        img_path = None
        for c in candidates:
            if os.path.isfile(c):
                img_path = c
                break
        if img_path is None:
            print("Image path not provided and image.jpeg not found. Usage: python c.py <image> [sigma]")
            return
        sigma_pos = 1
    sigma = 1.5
    if len(sys.argv) > sigma_pos:
        try:
            sigma = float(sys.argv[sigma_pos])
        except ValueError:
            pass
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    ensure_dir(out_dir)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load", img_path)
        return

    log_a = log_variant_a(img, sigma)
    log_b = log_variant_b(img, sigma)
    log_c = log_variant_c(img, sigma)

    cv2.imwrite(os.path.join(out_dir, 'lg_f.jpg'), log_a)
    cv2.imwrite(os.path.join(out_dir, 'l_gf.jpg'), log_b)
    cv2.imwrite(os.path.join(out_dir, 'glf.jpg'), log_c)
    print("Saved LoG variants (A,B,C) to", out_dir)

    try:
        labels = ['(L*G)*f', 'L*(G*f)', 'G*(L*f)']
        montage = montage_with_labels([log_a, log_b, log_c], labels, gap=25, top_margin=40, font_scale=0.7)
        cv2.imwrite(os.path.join(out_dir, 'log_combined_labeled.jpg'), montage)
        print("Saved labeled montage to", os.path.join(out_dir, 'log_combined_labeled.jpg'))
    except Exception as e:
        print("Could not create labeled montage:", e)

if __name__ == "__main__":
    main()
