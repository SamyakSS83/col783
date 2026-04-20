import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def ensure_dir(path: str) -> str:
    if not path or path.strip() == "":
        return os.getcwd()
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path

def load_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cv2.imread failed for {path}")
    return img

def save_image(path: str, img: np.ndarray) -> None:
    if img.dtype != np.uint8:
        img_to_save = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_to_save = img
    cv2.imwrite(path, img_to_save)

def plot_and_save_histogram(img: np.ndarray, out_path: str, title: str = None) -> None:
    plt.figure(figsize=(6,3))
    plt.hist(img.ravel(), bins=256, range=(0,255))
    if title:
        plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def manual_threshold(img: np.ndarray, thresh: int) -> np.ndarray:
    # Simple binary threshold (paper ~ white, text ~ black)
    _, out = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return out

def linear_transform(img: np.ndarray, a: float, b: float) -> np.ndarray:
    # Apply T(r) = a*r + b and clip to [0,255].
    out = a * img.astype(np.float32) + b
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def auto_linear_from_histogram(img: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> Tuple[np.ndarray, float, float]:
    if img.size == 0:
        raise ValueError("Empty image")
    low = np.percentile(img, low_pct)
    high = np.percentile(img, high_pct)
    if high <= low:
        # fallback to min/max
        low = float(img.min())
        high = float(img.max())
        if high == low:
            # constant image
            return img.copy(), 1.0, 0.0
    a = 255.0 / (high - low)
    b = -a * low
    out = linear_transform(img, a, b)
    return out, a, b

def gamma_transform(img: np.ndarray, gamma: float, eps: float = 1e-6) -> np.ndarray:
    norm = img.astype(np.float32) / 255.0
    out = np.power(norm + eps, gamma)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

def histogram_equalization(img: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(img)

def process_image(in_path: str, out_dir: str) -> None:
    out_dir = ensure_dir(out_dir)
    base_name = os.path.splitext(os.path.basename(in_path))[0]

    img_gray = load_image_gray(in_path)
    save_image(os.path.join(out_dir, f"{base_name}_orig_gray.png"), img_gray)
    plot_and_save_histogram(img_gray, os.path.join(out_dir, f"{base_name}_hist_orig.png"), "Original histogram")

    # 1) Manual thresholding
    manual_thresh_value = 150
    thr_manual = manual_threshold(img_gray, manual_thresh_value)
    save_image(os.path.join(out_dir, f"{base_name}_manual_threshold_{manual_thresh_value}.png"), thr_manual)
    plot_and_save_histogram(thr_manual, os.path.join(out_dir, f"{base_name}_hist_thr_manual.png"),
                            f"Manual threshold {manual_thresh_value}")

    # 2) Linear transform manually chosen
    a_manual = 1.9
    b_manual = -300.0
    # a_manual = 0.5
    # b_manual = -40.0
    lin_manual = linear_transform(img_gray, a_manual, b_manual)
    save_image(os.path.join(out_dir, f"{base_name}_linear_manual_a{a_manual}_b{int(b_manual)}.png"), lin_manual)
    plot_and_save_histogram(lin_manual, os.path.join(out_dir, f"{base_name}_hist_lin_manual.png"),
                            f"Linear manual a={a_manual}, b={b_manual}")

    # 3) Automatic linear transform based on histogram percentiles
    auto_out, a_auto, b_auto = auto_linear_from_histogram(img_gray, low_pct=1.0, high_pct=99.0)
    save_image(os.path.join(out_dir, f"{base_name}_linear_auto_pct2_98.png"), auto_out)
    plot_and_save_histogram(auto_out, os.path.join(out_dir, f"{base_name}_hist_lin_auto.png"),
                            f"Auto linear a={a_auto:.3f}, b={b_auto:.1f}")

    # 4) Gamma transforms
    gamma_values = [0.6, 1.0, 1.6]
    for g in gamma_values:
        out_g = gamma_transform(img_gray, g)
        save_image(os.path.join(out_dir, f"{base_name}_gamma_{g:.2f}.png"), out_g)
        plot_and_save_histogram(out_g, os.path.join(out_dir, f"{base_name}_hist_gamma_{g:.2f}.png"),
                                f"Gamma {g}")

    # 5) Histogram equalization
    heq = histogram_equalization(img_gray)
    save_image(os.path.join(out_dir, f"{base_name}_histeq.png"), heq)
    plot_and_save_histogram(heq, os.path.join(out_dir, f"{base_name}_hist_histeq.png"), "Histogram Equalized")

    print(f"Saved results to: {out_dir}")
    print(f"Auto linear parameters: a={a_auto:.4f}, b={b_auto:.2f}")
    print("Manual threshold used:", manual_thresh_value)
    print("Manual linear used: a=", a_manual, "b=", b_manual)
    print("Gamma values:", gamma_values)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python intensity.py <input_image> [output_dir]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) == 3 else os.path.join(os.path.dirname(__file__), "results")
    ensure_dir(out_dir)

    if not os.path.isfile(in_path):
        raise RuntimeError(f"Input image not found: {in_path}")

    process_image(in_path, out_dir)

if __name__ == "__main__":
    main()
