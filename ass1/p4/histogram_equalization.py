import os
import sys
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

def fast_linear_interpolate(x_values, y_values, query_points):
    # Vectorized linear interpolation using numpy.
    q = query_points.reshape(-1)
    idx = np.searchsorted(x_values, q, side='right') - 1
    idx = np.clip(idx, 0, len(x_values) - 2)

    x1 = x_values[idx]
    x2 = x_values[idx + 1]
    y1 = y_values[idx]
    y2 = y_values[idx + 1]

    dx = x2 - x1
    dx = np.where(dx == 0, 1, dx)

    t = (q - x1) / dx
    out = y1 + t * (y2 - y1)

    out = np.where(q <= x_values[0], y_values[0], out)
    out = np.where(q >= x_values[-1], y_values[-1], out)
    return out.reshape(query_points.shape)

def _equalize_single_channel(channel, a, b):
    # Flatten and sort to build empirical CDF
    flat = channel.reshape(-1)
    n = flat.size

    sorted_vals = np.sort(flat)
    ranks = np.arange(1, n + 1, dtype=np.float64) / n  # F(x)

    # Map CDF to uniform range [a, b]
    mapped = a + (b - a) * ranks

    # Use unique intensities for a compact, stable mapping
    uniq_vals, uniq_idx = np.unique(sorted_vals, return_index=True)
    uniq_mapped = mapped[uniq_idx]

    # Interpolate back for every pixel
    return fast_linear_interpolate(uniq_vals, uniq_mapped, channel)

def equalize_histogram_hdr(image, a=0.0, b=256.0, use_threading=True):
    # Histogram equalization for real-valued HDR images without quantization.
    if image.ndim == 3:
        if use_threading:
            with ThreadPoolExecutor(max_workers=image.shape[2]) as pool:
                chans = [image[:, :, c] for c in range(image.shape[2])]
                eq = list(pool.map(lambda ch: _equalize_single_channel(ch, a, b), chans))
            return np.stack(eq, axis=2), None
        else:
            eq = [_equalize_single_channel(image[:, :, c], a, b) for c in range(image.shape[2])]
            return np.stack(eq, axis=2), None
    else:
        eq = _equalize_single_channel(image, a, b)
        return eq, None

def _ensure_results_dir(path="results"):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path

def save_image_pair(original, equalized, prefix="hdr"):
    outdir = _ensure_results_dir()

    # Prepare 8-bit saves
    if original.ndim == 3:
        orig_8u = np.clip((original / (np.max(original) + 1e-12)) * 255, 0, 255).astype(np.uint8)
        eq_8u = np.clip((equalized / 256.0) * 255, 0, 255).astype(np.uint8)
    else:
        orig_8u = np.clip((original / (np.max(original) + 1e-12)) * 255, 0, 255).astype(np.uint8)
        eq_8u = np.clip(equalized, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(outdir, f"{prefix}_original.jpg"), orig_8u)
    cv2.imwrite(os.path.join(outdir, f"{prefix}_equalized.jpg"), eq_8u)

def main():
    if len(sys.argv) != 2:
        print("Usage: python try.py <hdr_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    hdr = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if hdr is None:
        print("Error: could not load", filename)
        sys.exit(1)

    # OpenCV loads BGR; equalization is content-agnostic, but save RGB for viewing
    hdr_rgb = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)

    # Equalize to [0, 256)
    eq_rgb, _ = equalize_histogram_hdr(hdr_rgb.astype(np.float32), 0.0, 256.0, use_threading=True)

    # Save outputs under results/
    save_image_pair(hdr_rgb, eq_rgb, prefix="hdr")
    print("Saved results to ./results (hdr_original.jpg, hdr_equalized.jpg)")

if __name__ == "__main__":
    main()