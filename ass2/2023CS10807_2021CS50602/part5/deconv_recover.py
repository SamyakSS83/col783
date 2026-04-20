import os
import sys
import math
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return (img.astype(np.float32) / 255.0).astype(np.float32)

def save_gray(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)

def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 1.0) -> float:
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10((max_val ** 2) / mse)

# Load PSF (from part2 outputs, minimal)
def load_large_psf(base_dir: str) -> np.ndarray:
    candidates = [
        os.path.join(base_dir, 'part2', 'psf_80.png'),
        os.path.join(base_dir, 'part2', 'psf_160.png'),
        os.path.join(base_dir, 'part2', 'h.png'),
    ]
    for c in candidates:
        if os.path.exists(c):
            k = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            if k is None:
                continue
            k = k.astype(np.float32)
            if k.sum() == 0:
                k[:] = 1.0
            k = k / k.sum()
            if k.shape[0] < 80:
                k = cv2.resize(k, (80, 80), interpolation=cv2.INTER_CUBIC)
                k = np.clip(k, 0, None)
                k /= k.sum()
            return k.astype(np.float32)
    raise FileNotFoundError("Could not locate PSF kernel in part2 (psf_80.png or h.png)")

# Blur in frequency domain (normalize AFTER embedding)
def blur_with_psf(img: np.ndarray, psf: np.ndarray):
    H, W = img.shape
    kh, kw = psf.shape
    pad = np.zeros_like(img, dtype=np.float32)
    sh = (H - kh) // 2
    sw = (W - kw) // 2
    pad[sh:sh + kh, sw:sw + kw] = psf.astype(np.float32)
    s = pad.sum()
    if s <= 0:
        raise ValueError("Embedded PSF has zero sum")
    pad /= s
    pad_c = np.fft.ifftshift(pad)
    F = np.fft.fft2(img).astype(np.complex64)
    Hf = np.fft.fft2(pad_c).astype(np.complex64)
    G = F * Hf
    g = np.fft.ifft2(G).real.astype(np.float32)
    del F, pad, pad_c, G
    gc.collect()
    return np.clip(g, 0.0, 1.0), Hf

# Gaussian noise generator
def gaussian_noise_for_target_psnr(img: np.ndarray, target_psnr: float, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    target_mse = (1.0 ** 2) / (10 ** (target_psnr / 10.0))
    sigma = math.sqrt(target_mse)
    noise = rng.normal(0.0, 1.0, img.shape).astype(np.float32)
    cur_std = noise.std()
    if cur_std > 0:
        noise *= (sigma / (cur_std + 1e-12))
    noisy_unclipped = img + noise
    cur_mse = np.mean((noisy_unclipped - img) ** 2)
    if cur_mse > 0:
        noisy_unclipped = img + (noisy_unclipped - img) * math.sqrt(target_mse / (cur_mse + 1e-12))
    noisy_clipped = np.clip(noisy_unclipped, 0.0, 1.0).astype(np.float32)
    actual = psnr(noisy_clipped, img)
    return noisy_unclipped.astype(np.float32), noisy_clipped.astype(np.float32), actual, sigma

# Estimate Sf from the noiseless blurred image (robust)
def estimate_Sf_from_blurred(blurred: np.ndarray, Hf: np.ndarray, smooth_ksize: int = 7, epsH: float = 1e-8):
    M, N = blurred.shape
    B = np.fft.fft2(blurred).astype(np.complex64)
    S_b = (np.abs(B) ** 2) / (M * N)   # float-ish
    del B
    gc.collect()

    # center and reduce precision to float32
    Sb_shift = np.fft.fftshift(S_b).astype(np.float32)
    # gaussian blur to reduce variance
    k = smooth_ksize if (smooth_ksize % 2 == 1) else (smooth_ksize + 1)
    Sb_blur = cv2.GaussianBlur(Sb_shift, (k, k), 0).astype(np.float32)
    Sb_un = np.fft.ifftshift(Sb_blur)

    Habs2 = (np.abs(Hf) ** 2).astype(np.float32)
    Sf_est = Sb_un / np.maximum(Habs2, epsH)

    # smoothing S_f estimate and robust clipping
    Sf_shift = np.fft.fftshift(Sf_est).astype(np.float32)
    Sf_shift_sm = cv2.GaussianBlur(Sf_shift, (k, k), 0).astype(np.float32)
    Sf_sm = np.fft.ifftshift(Sf_shift_sm)

    p95 = np.percentile(Sf_sm, 95)
    if not np.isfinite(p95) or p95 <= 0:
        p95 = np.max(Sf_sm)
    Sf_sm_clipped = np.minimum(Sf_sm, p95).astype(np.float32)
    Sf_sm_clipped = np.maximum(Sf_sm_clipped, 1e-10)

    del S_b, Sb_shift, Sb_blur, Sb_un, Sf_shift, Sf_shift_sm
    gc.collect()
    return Sf_sm_clipped.astype(np.float32)

# Adaptive Wiener using Sf estimate from blurred (noiseless)
def adaptive_wiener_using_blurred(g_proc: np.ndarray, Hf: np.ndarray, blurred: np.ndarray, sigma2: float,
                                  smooth_ksize: int = 7):
    Sf_est = estimate_Sf_from_blurred(blurred, Hf, smooth_ksize=smooth_ksize)
    K_map = (sigma2 / (Sf_est + 1e-20)).astype(np.float32)
    K_map = np.clip(K_map, 1e-10, 1e8).astype(np.float32)

    Habs2 = (np.abs(Hf) ** 2).astype(np.float32)
    h_thresh = np.percentile(Habs2, 5)
    mask_small = (Habs2 < h_thresh)
    if mask_small.any():
        # raise K_map where H small so the Wiener filter suppresses those frequencies
        p99 = float(np.percentile(K_map, 99))
        K_map[mask_small] = np.maximum(K_map[mask_small], p99 * 10.0)

    # now apply Wiener using K_map
    G = np.fft.fft2(g_proc).astype(np.complex64)
    Hc = np.conjugate(Hf)
    denom = (Habs2 + K_map).astype(np.float32)
    denom_c = denom.astype(np.complex64)
    W = (Hc.astype(np.complex64) / (denom_c + 1e-20))
    Fhat = W * G
    rec = np.fft.ifft2(Fhat).real.astype(np.float32)
    rec = np.clip(rec, 0.0, 1.0)

    del G, Hc, denom, denom_c, W, Fhat, Sf_est, Habs2
    gc.collect()
    return rec, K_map

# streaming band saves (no big lists)
def save_lowpass_and_bands_stream(img: np.ndarray, outdir: str, prefix: str, max_levels: int | None = None):
    M, N = img.shape
    if max_levels is None:
        max_levels = int(math.floor(math.log2(min(M, N)))) if min(M, N) >= 1 else 0
    F = np.fft.fftshift(np.fft.fft2(img)).astype(np.complex64)
    li_prev = None
    for i in range(max_levels + 1):
        side = 2 ** i
        if side > min(M, N):
            break
        mask = np.zeros((M, N), dtype=np.float32)
        y0 = M // 2 - side // 2
        x0 = N // 2 - side // 2
        mask[y0:y0 + side, x0:x0 + side] = 1.0
        Fi = F * mask.astype(np.complex64)
        li = np.fft.ifft2(np.fft.ifftshift(Fi)).real.astype(np.float32)
        if i > 0:
            save_gray(os.path.join(outdir, f"{prefix}_l{i}.png"), np.clip(li, 0.0, 1.0))
        if li_prev is not None:
            b = li - li_prev
            disp = 0.5 + b
            disp = np.clip(disp, 0.0, 1.0)
            save_gray(os.path.join(outdir, f"{prefix}_b{i}.png"), disp)
            del b, disp
        li_prev = li
        del Fi
        gc.collect()
    del F, li_prev, mask
    gc.collect()

# Constant-K Wiener sweep (reduced grid)
def run_wiener_sweep(g: np.ndarray, f_orig: np.ndarray, Hf: np.ndarray, K_values: np.ndarray, out_plot: str, K_mark: float):
    results = []
    best_img = None
    best_psnr = -1
    for K in K_values:
        G = np.fft.fft2(g).astype(np.complex64)
        Hc = np.conjugate(Hf)
        denom = (np.abs(Hf) ** 2 + float(K))
        denom_c = denom.astype(np.complex64)
        Fhat = (Hc.astype(np.complex64) / denom_c) * G
        rec = np.fft.ifft2(Fhat).real.astype(np.float32)
        rec = np.clip(rec, 0.0, 1.0)
        p = psnr(rec, f_orig)
        results.append((K, p))
        if p > best_psnr:
            best_psnr = p
            best_img = rec.copy()
        del G, Hc, denom, denom_c, Fhat, rec
        gc.collect()
    Ks = [k for k, _ in results]
    Ps = [p for _, p in results]
    plt.figure()
    plt.semilogx(Ks, Ps, marker='o')
    plt.axvline(K_mark, linestyle='--', label=f'K_est={K_mark:.3e}')
    plt.xlabel('K')
    plt.ylabel('PSNR (dB)')
    plt.title('Wiener Filter PSNR vs K')
    plt.legend()
    plt.grid(True, which='both', ls=':')
    d = os.path.dirname(out_plot)
    if d:
        os.makedirs(d, exist_ok=True)
    plt.savefig(out_plot)
    plt.close()
    gc.collect()
    return best_img, best_psnr, results

def compute_global_Sf_prior(f: np.ndarray):
    """Flat-spectrum prior S_f = ||f||^2 / (M*N)"""
    M, N = f.shape
    Sf_prior = np.ones((M, N), dtype=np.float32) * (np.sum(f**2) / (M * N))
    return Sf_prior

def build_K_map_from_Sf(Sf_final: np.ndarray, sigma2: float, kmin=1e-12, kmax=1e6):
    """Compute K_map = sigma2 / Sf_final, clamp dynamic range, return float32."""
    K_map = (sigma2 / (Sf_final + 1e-20)).astype(np.float32)
    K_map = np.clip(K_map, kmin, kmax)
    return K_map

def tune_and_apply_adaptive(g_proc, Hf, blurred, f, sigma2,
                            smooth_ksizes=(3,5,9,15),
                            betas=(0.0,0.25,0.5,0.75,1.0),
                            kmax=1e5):
    best_psnr = -1.0
    best_rec = None
    best_K = None
    best_params = None

    # Precompute Sf_local for each smoothing size (cache to avoid recompute)
    Sf_local_cache = {}
    for ks in smooth_ksizes:
        Sf_local_cache[ks] = estimate_Sf_from_blurred(blurred, Hf, smooth_ksize=ks, epsH=1e-8)

    Sf_prior = compute_global_Sf_prior(f)

    for ks in smooth_ksizes:
        Sf_local = Sf_local_cache[ks]
        for beta in betas:
            Sf_final = beta * Sf_local + (1.0 - beta) * Sf_prior
            # optional small smoothing on Sf_final (reduce speckle)
            Sf_final_shift = np.fft.fftshift(Sf_final).astype(np.float32)
            Sf_final_sm = cv2.GaussianBlur(Sf_final_shift, (3,3), 0)
            Sf_final = np.fft.ifftshift(Sf_final_sm).astype(np.float32)

            K_map = build_K_map_from_Sf(Sf_final, sigma2, kmin=1e-12, kmax=kmax)

            # further clamp where H is tiny (avoid amplifying)
            Habs2 = (np.abs(Hf)**2).astype(np.float32)
            h_thresh = np.percentile(Habs2, 3)
            mask_small = (Habs2 < h_thresh)
            if mask_small.any():
                K_map[mask_small] = np.maximum(K_map[mask_small], np.percentile(K_map, 95)*5.0)

            # optional final smooth of K_map
            Kshift = np.fft.fftshift(K_map).astype(np.float32)
            Kshift_sm = cv2.GaussianBlur(Kshift, (3,3), 0)
            K_map = np.fft.ifftshift(Kshift_sm).astype(np.float32)

            # apply Wiener with this K_map
            rec = adaptive_wiener_using_blurred(g_proc, Hf, blurred, sigma2, smooth_ksize=ks)[0]  # this currently computes Sf internally; we'll instead use direct Wiener below

            # But to use our K_map we need a function that accepts K_map. Implement inline:
            G = np.fft.fft2(g_proc).astype(np.complex64)
            Hc = np.conjugate(Hf)
            denom = (np.abs(Hf)**2 + K_map).astype(np.float32)
            denom_c = denom.astype(np.complex64)
            W = (Hc.astype(np.complex64) / (denom_c + 1e-20))
            Fhat = W * G
            rec = np.fft.ifft2(Fhat).real.astype(np.float32)
            rec = np.clip(rec, 0.0, 1.0)
            p = psnr(rec, f)
            del G, Hc, denom, denom_c, W, Fhat
            gc.collect()

            if p > best_psnr:
                best_psnr = p
                best_rec = rec.copy()
                best_K = K_map.copy()
                best_params = {'smooth_ks': ks, 'beta': beta, 'kmax': kmax}

    return best_rec, best_K, best_params

def main():
    if len(sys.argv) != 3:
        print("Usage: python deconv_recover.py <input_image> <outdir>")
        sys.exit(1)
    in_path = sys.argv[1]
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    f = load_gray(in_path)
    save_gray(os.path.join(outdir, 'f_orig.png'), f)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    psf = load_large_psf(root_dir)
    blurred, Hf = blur_with_psf(f, psf)
    save_gray(os.path.join(outdir, 'f_blurred.png'), blurred)

    noise_levels = [20, 30, 10]
    f_energy = np.sum(f ** 2)
    M, N = f.shape

    for target in noise_levels:
        print(f"\n--- target PSNR {target} dB ---")
        noisy_unclipped, noisy_clipped, actual_psnr, sigma = gaussian_noise_for_target_psnr(blurred, target)
        save_gray(os.path.join(outdir, f'g_psnr{target}_actual{actual_psnr:.2f}.png'), noisy_clipped)
        g_proc = noisy_unclipped
        g_display = noisy_clipped

        if target == noise_levels[0]:
            save_lowpass_and_bands_stream(f, os.path.join(outdir, 'bands_f'), 'f')
        save_lowpass_and_bands_stream(g_display, os.path.join(outdir, f'bands_g_{target}'), f'g{target}')

        try:
            eps = 1e-12
            G = np.fft.fft2(g_proc).astype(np.complex64)
            Fhat = np.where(np.abs(Hf) > eps, G / Hf, 0.0)
            inv = np.fft.ifft2(Fhat).real.astype(np.float32)
            inv = np.clip(inv, 0.0, 1.0)
            save_gray(os.path.join(outdir, f'inverse_{target}.png'), inv)
            save_lowpass_and_bands_stream(inv, os.path.join(outdir, f'bands_inv_{target}'), f'inv{target}')
            del G, Fhat, inv
            gc.collect()
        except Exception as e:
            print("Inverse failed:", e)

        # estimate K constant
        sigma2 = sigma ** 2
        S_f_const = f_energy / (M * N)
        K_est = sigma2 / (S_f_const + 1e-20)

        # constant-K Wiener sweep
        Ks1 = np.logspace(-6, 1, 30)
        Ks2 = K_est * np.logspace(-2, 2, 15)
        K_values = np.unique(np.concatenate([Ks1, Ks2]))
        best_img, best_psnr, results = run_wiener_sweep(g_proc, f, Hf, K_values,
                                                       os.path.join(outdir, f'wiener_constant_psnr_{target}.png'),
                                                       K_est)
        save_gray(os.path.join(outdir, f'wiener_constant_best_{target}_{best_psnr:.2f}.png'), best_img)
        save_lowpass_and_bands_stream(best_img, os.path.join(outdir, f'bands_wiener_constant_{target}'), f'wc{target}')
        del best_img
        gc.collect()

        # Adaptive Wiener using noiseless blurred to estimate S_f
        # rec_ad, K_map = adaptive_wiener_using_blurred(g_proc, Hf, blurred, sigma2, smooth_ksize=7)
        best_rec, best_K, best_params = tune_and_apply_adaptive(g_proc, Hf, blurred, f, sigma2,
                                                       smooth_ksizes=(3,5,9),
                                                       betas=(0.0,0.25,0.5,0.75,1.0),
                                                       kmax=1e5)
        adaptive_psnr = psnr(best_rec, f)
        print("Adaptive best params:", best_params, "PSNR:", adaptive_psnr)

        print(f"PSNR(g_display,f) = {psnr(g_display,f):.4f} dB, PSNR(adaptive_rec,f) = {adaptive_psnr:.4f} dB")
        save_gray(os.path.join(outdir, f'wiener_adaptive_{target}_{adaptive_psnr:.2f}.png'), best_rec)
        save_lowpass_and_bands_stream(best_rec, os.path.join(outdir, f'bands_wiener_adaptive_{target}'), f'wa{target}')

        ratio_vis = np.log10(best_K + 1e-12)
        ratio_vis_norm = (ratio_vis - ratio_vis.min()) / (ratio_vis.max() - ratio_vis.min() + 1e-12)
        save_gray(os.path.join(outdir, f'wiener_adaptive_ratio_{target}.png'), ratio_vis_norm)

        with open(os.path.join(outdir, f'wiener_results_{target}.txt'), 'w') as fh:
            fh.write(f"Target noise PSNR: {target}\n")
            fh.write(f"Actual noisy PSNR (clipped, blurred vs noisy): {actual_psnr:.4f}\n")
            fh.write(f"K_est (const): {K_est:.6e}\n")
            fh.write(f"Adaptive Wiener PSNR: {adaptive_psnr:.4f}\n")
            fh.write("K_constant,PSNR_constant(dB)\n")
            for K, P in results:
                fh.write(f"{K:.6e},{P:.6f}\n")

        if adaptive_psnr <= psnr(g_display, f) + 1e-9:
            print("Warning: adaptive Wiener did not improve PSNR. Saved debug images (fail_*)")
            save_gray(os.path.join(outdir, f'fail_g_{target}.png'), g_display)
            save_gray(os.path.join(outdir, f'fail_rec_{target}.png'), best_rec)
            diff = np.clip(best_rec - g_display + 0.5, 0.0, 1.0)
            save_gray(os.path.join(outdir, f'fail_diff_{target}.png'), diff)

        del g_proc, g_display, best_rec, best_K
        gc.collect()

    print("Done. Results in", outdir)

if __name__ == '__main__':
    main()
