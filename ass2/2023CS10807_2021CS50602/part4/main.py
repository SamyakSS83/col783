# part4/main.py
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def to_float01_from_uint8(img_uint8):
    return img_uint8.astype(np.float32) / 255.0

def to_uint8_from_float01(img_float):
    return np.clip((img_float * 255.0), 0, 255).astype(np.uint8)

def pnsr(img, ref, max_val=1.0):
    mse = np.mean((img - ref) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((max_val ** 2) / mse)

def add_uni_noise(img, target_pnsr, max_val=1.0):
    ref = img.astype(np.float32)
    target_mse = (max_val ** 2) / (10 ** (target_pnsr / 10.0))
    a = np.sqrt(3.0 * target_mse)
    noise = np.random.uniform(-a, a, img.shape).astype(np.float32)
    noisy = ref + noise

    cur_mse = np.mean((noisy - ref) ** 2)
    if cur_mse > 0:
        scale = np.sqrt(target_mse / (cur_mse + 1e-12))
        noisy = ref + (noisy - ref) * scale

    noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)
    actual = pnsr(noisy, ref)
    print(f"[add_uni_noise] target={target_pnsr} dB, actual={actual:.4f} dB")
    return noisy, actual

def add_gauss_noise(img, target_pnsr, max_val=1.0):
    ref = img.astype(np.float32)
    target_mse = (max_val ** 2) / (10 ** (target_pnsr / 10.0))
    sigma = np.sqrt(target_mse)
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    noisy = ref + noise

    cur_mse = np.mean((noisy - ref) ** 2)
    if cur_mse > 0:
        scale = np.sqrt(target_mse / (cur_mse + 1e-12))
        noisy = ref + (noisy - ref) * scale

    noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)
    actual = pnsr(noisy, ref)
    print(f"[add_gauss_noise] target={target_pnsr} dB, actual={actual:.4f} dB")
    return noisy, actual

def salt_pepper_prob_for_pnsr(img, target_pnsr, max_val=1.0):
    img = img.astype(np.float32)
    N = img.size
    E0 = np.sum(img ** 2) / N
    Emax = np.sum((img - max_val) ** 2) / N
    mse_target = (max_val ** 2) / (10 ** (target_pnsr / 10.0))
    p = (2 * mse_target) / (E0 + Emax + 1e-12)
    return float(np.clip(p, 0.0, 1.0))

def add_salt_pepper_noise(img, target_pnsr, max_val=1.0):
    ref = img.astype(np.float32)
    p = salt_pepper_prob_for_pnsr(ref, target_pnsr, max_val)
    rnd = np.random.rand(*img.shape).astype(np.float32)
    noisy = ref.copy()
    noisy[rnd < (p / 2.0)] = 0.0
    noisy[rnd > (1.0 - p / 2.0)] = max_val
    noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)
    actual = pnsr(noisy, ref)
    print(f"[add_salt_pepper_noise] target={target_pnsr} dB, p={p:.6f}, actual={actual:.4f} dB")
    return noisy, actual

def pad_reflect(img, pad_size):
    return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

def mean_filter(img, w):
    if w == 1:
        return img.copy()
    out = ndimage.uniform_filter(img.astype(np.float32), size=w, mode='reflect')
    return np.clip(out, 0.0, 1.0)

def median_filter(img, w):
    if w == 1:
        return img.copy()
    out = ndimage.median_filter(img.astype(np.float32), size=w, mode='reflect')
    return np.clip(out, 0.0, 1.0)

def save_and_eval(noisy_float01, ref_float01, outpath):
    u8 = to_uint8_from_float01(noisy_float01)
    cv2.imwrite(outpath, u8)
    return pnsr(noisy_float01, ref_float01)

def plot_pnsr_curves(pnsr_mean, pnsr_median, noise_name, outdir):
    ws = [w for w, _ in pnsr_mean]
    vals_mean = [v for _, v in pnsr_mean]
    vals_median = [v for _, v in pnsr_median]

    plt.figure()
    plt.plot(ws, vals_mean, marker='o', label="Mean Filter")
    plt.plot(ws, vals_median, marker='s', label="Median Filter")
    plt.xlabel("Filter width w")
    plt.ylabel("pnsr (dB)")
    plt.title(f"pnsr vs w ({noise_name})")
    plt.legend()
    plt.grid(True)
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"{noise_name}_pnsr_plot.png"))
    plt.close()

def process_image(img_float01, ref_float01, outdir, noise_name, noise_fn):
    noisy, actual = noise_fn(ref_float01, target_pnsr=20)
    cv2.imwrite(os.path.join(outdir, f"{noise_name}_noisy.png"), to_uint8_from_float01(noisy))

    pnsr_results_mean, pnsr_results_median = [], []
    max_w = 11
    for w in range(1, max_w+1, 2):
        mean_out = mean_filter(noisy, w)
        median_out = median_filter(noisy, w)
        pnsr_results_mean.append((w, pnsr(mean_out, ref_float01)))
        pnsr_results_median.append((w, pnsr(median_out, ref_float01)))
        cv2.imwrite(os.path.join(outdir, f"{noise_name}_mean_{w}.png"), to_uint8_from_float01(mean_out))
        cv2.imwrite(os.path.join(outdir, f"{noise_name}_median_{w}.png"), to_uint8_from_float01(median_out))

    plot_pnsr_curves(pnsr_results_mean, pnsr_results_median, noise_name, outdir)
    return pnsr_results_mean, pnsr_results_median

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <input natural image> <output dir>")
        sys.exit(1)

    img_path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    natural_u8 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if natural_u8 is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    natural = to_float01_from_uint8(natural_u8)
    constant = np.ones_like(natural, dtype=np.float32) * 0.5

    noise_models = [
        ("uniform", add_uni_noise),
        ("gaussian", add_gauss_noise),
        ("saltpepper", add_salt_pepper_noise),
    ]

    for name, fn in noise_models:
        print(f"\nProcessing constant image with {name} noise...")
        process_image(constant, constant, outdir, f"c_{name}", fn)

        print(f"Processing natural image with {name} noise...")
        process_image(natural, natural, outdir, f"f_{name}", fn)

if __name__ == "__main__":
    main()
