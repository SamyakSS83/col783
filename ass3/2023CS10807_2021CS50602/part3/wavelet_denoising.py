import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class WaveletDenoising:


    def __init__(self):
        sqrt3 = np.sqrt(3.0)
        sqrt2 = np.sqrt(2.0)

        
        self.db2_h = np.array([
            (1.0 + sqrt3) / (4.0 * sqrt2),
            (3.0 + sqrt3) / (4.0 * sqrt2),
            (3.0 - sqrt3) / (4.0 * sqrt2),
            (1.0 - sqrt3) / (4.0 * sqrt2)
        ], dtype=float)

        
        self.db2_g = np.array([
            (1.0 - sqrt3) / (4.0 * sqrt2),
            -(3.0 - sqrt3) / (4.0 * sqrt2),
            (3.0 + sqrt3) / (4.0 * sqrt2),
            -(1.0 + sqrt3) / (4.0 * sqrt2)
        ], dtype=float)

        
        self.db2_h_rec = self.db2_h[::-1].copy()
        self.db2_g_rec = self.db2_g[::-1].copy()

    
    def haar_transform_1d(self, signal):
        n = len(signal)
        if n == 1:
            return signal.copy()
        half = n // 2
        res = np.zeros(n, dtype=float)
        res[:half] = (signal[::2] + signal[1::2]) / np.sqrt(2.0)
        res[half:] = (signal[::2] - signal[1::2]) / np.sqrt(2.0)
        return res

    def haar_inverse_1d(self, coeffs):
        n = len(coeffs)
        if n == 1:
            return coeffs.copy()
        half = n // 2
        approx = coeffs[:half]
        detail = coeffs[half:]
        res = np.zeros(n, dtype=float)
        res[::2] = (approx + detail) / np.sqrt(2.0)
        res[1::2] = (approx - detail) / np.sqrt(2.0)
        return res

    def multilevel_haar_transform(self, image, levels):
        result = image.copy().astype(float)
        h, w = result.shape
        for level in range(levels):
            size = h // (2 ** level)
            if size < 2:
                break
            sub = result[:size, :size].copy()
            
            for i in range(size):
                sub[i, :] = self.haar_transform_1d(sub[i, :])
            
            for j in range(size):
                sub[:, j] = self.haar_transform_1d(sub[:, j])
            result[:size, :size] = sub
        return result

    def multilevel_haar_inverse(self, coeffs, levels):
        result = coeffs.copy().astype(float)
        h, w = result.shape
        for level in range(levels - 1, -1, -1):
            size = h // (2 ** level)
            if size < 2:
                continue
            sub = result[:size, :size].copy()
            for j in range(size):
                sub[:, j] = self.haar_inverse_1d(sub[:, j])
            for i in range(size):
                sub[i, :] = self.haar_inverse_1d(sub[i, :])
            result[:size, :size] = sub
        return result

    
    def db2_transform_1d(self, signal):
        """
        Forward db2 1D transform: periodic extension, convolution then downsample.
        output: [approx(0..n/2-1), detail(0..n/2-1)] length n
        """
        n = len(signal)
        if n == 1:
            return signal.copy()
        half = n // 2
        
        approx = np.zeros(half, dtype=float)
        detail = np.zeros(half, dtype=float)
        L = len(self.db2_h)  
        
        for i in range(half):
            idx = 2 * i
            
            s_approx = 0.0
            s_detail = 0.0
            for k in range(L):
                s_approx += self.db2_h[k] * signal[(idx + k) % n]
                s_detail += self.db2_g[k] * signal[(idx + k) % n]
            approx[i] = s_approx
            detail[i] = s_detail
        return np.concatenate([approx, detail])

    def db2_inverse_1d(self, coeffs):
        n = len(coeffs)
        if n == 1:
            return coeffs.copy()
        half = n // 2
        approx = coeffs[:half]
        detail = coeffs[half:]
        
        approx_up = np.zeros(n, dtype=float)
        detail_up = np.zeros(n, dtype=float)
        approx_up[0: n: 2] = approx  
        detail_up[0: n: 2] = detail  
        L = len(self.db2_h_rec)  
        res = np.zeros(n, dtype=float)
        
        for i in range(n):
            val = 0.0
            
            for k in range(L):
                
                j = (i - k) % n
                val += self.db2_h_rec[k] * approx_up[j]
                val += self.db2_g_rec[k] * detail_up[j]
            res[i] = val
        return res

    def multilevel_db2_transform(self, image, levels):
        result = image.copy().astype(float)
        h, w = result.shape
        for level in range(levels):
            size = h // (2 ** level)
            if size < 2:
                break
            sub = result[:size, :size].copy()
            
            for i in range(size):
                sub[i, :] = self.db2_transform_1d(sub[i, :])
            
            for j in range(size):
                sub[:, j] = self.db2_transform_1d(sub[:, j])
            result[:size, :size] = sub
        return result

    def multilevel_db2_inverse(self, coeffs, levels):
        result = coeffs.copy().astype(float)
        h, w = result.shape
        for level in range(levels - 1, -1, -1):
            size = h // (2 ** level)
            if size < 2:
                continue
            sub = result[:size, :size].copy()
            for j in range(size):
                sub[:, j] = self.db2_inverse_1d(sub[:, j])
            for i in range(size):
                sub[i, :] = self.db2_inverse_1d(sub[i, :])
            result[:size, :size] = sub
        return result

    
    def estimate_noise_sigma(self, coeffs, levels):
        detail = self.get_detail_coefficients(coeffs, levels)
        if detail.size == 0:
            return 1.0
        mad = np.median(np.abs(detail - np.median(detail)))
        sigma = mad / 0.6745 if mad > 0 else np.std(detail)  
        return sigma

    def universal_threshold(self, coeffs, levels):
        detail = self.get_detail_coefficients(coeffs, levels)
        N = max(1, detail.size)
        sigma = self.estimate_noise_sigma(coeffs, levels)
        return sigma * np.sqrt(2.0 * np.log(N + 1.0))

    def hard_threshold(self, coeffs, threshold, levels):
        result = coeffs.copy()
        h, w = result.shape
        coarse_size = h // (2 ** levels)
        
        mask = np.ones_like(result, dtype=bool)
        mask[:coarse_size, :coarse_size] = False  
        result[mask & (np.abs(result) < threshold)] = 0.0
        return result

    def soft_threshold(self, coeffs, threshold, levels):
        result = coeffs.copy()
        h, w = result.shape
        coarse_size = h // (2 ** levels)
        for i in range(h):
            for j in range(w):
                if i >= coarse_size or j >= coarse_size:
                    val = result[i, j]
                    if np.abs(val) <= threshold:
                        result[i, j] = 0.0
                    else:
                        result[i, j] = np.sign(val) * (np.abs(val) - threshold)
        return result

    
    def get_detail_coefficients(self, coeffs, levels):
        h, w = coeffs.shape
        coarse_size = h // (2 ** levels)
        
        details = []
        for i in range(h):
            for j in range(w):
                if i >= coarse_size or j >= coarse_size:
                    details.append(coeffs[i, j])
        return np.array(details, dtype=float)

    def visualize_coefficients(self, coeffs, levels, title="Wavelet Coefficients", out_dir="./results"):
        os.makedirs(out_dir, exist_ok=True)
        vis = coeffs.copy()
        h, w = vis.shape
        coarse_size = h // (2 ** levels)
        
        for i in range(h):
            for j in range(w):
                if i >= coarse_size or j >= coarse_size:
                    vis[i, j] = vis[i, j] + 128.0
        vis = np.clip(vis, 0, 255)
        plt.figure(figsize=(6,6))
        plt.imshow(vis, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{title.replace(' ', '_')}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        return vis

    def calculate_psnr(self, original, reconstructed):
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20.0 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def add_gaussian_noise(self, image, target_psnr=20):
        image = image.astype(float)
        max_pixel = 255.0
        sigma = max_pixel / (10.0 ** (target_psnr / 20.0))
        noise = np.random.normal(0.0, sigma, image.shape)
        noisy = image + noise
        noisy = np.clip(noisy, 0.0, 255.0)
        return noisy


def main():
    
    outdir = "./results"
    os.makedirs(outdir, exist_ok=True)

    print("Loading image.")
    image_path = "apple.png"  
    img = Image.open(image_path).convert('L')
    img = img.resize((512, 512))
    original = np.array(img).astype(float)
    denoiser = WaveletDenoising()

    print("Adding Gaussian noise (target PSNR = 20 dB).")
    noisy = denoiser.add_gaussian_noise(original, target_psnr=20)
    print("Actual noisy PSNR:", denoiser.calculate_psnr(original, noisy))

    levels = 4

    
    print("Computing Haar transform...")
    haar_orig = denoiser.multilevel_haar_transform(original, levels)
    haar_noisy = denoiser.multilevel_haar_transform(noisy, levels)
    denoiser.visualize_coefficients(haar_orig, levels, "Haar_Original_Coefficients", outdir)
    denoiser.visualize_coefficients(haar_noisy, levels, "Haar_Noisy_Coefficients", outdir)

    
    t_haar = denoiser.universal_threshold(haar_noisy, levels)
    print(f"Haar automatic universal threshold: {t_haar:.3f}")

    hard = denoiser.hard_threshold(haar_noisy, t_haar, levels)
    soft = denoiser.soft_threshold(haar_noisy, t_haar, levels)
    hard_rec = np.clip(denoiser.multilevel_haar_inverse(hard, levels), 0, 255)
    soft_rec = np.clip(denoiser.multilevel_haar_inverse(soft, levels), 0, 255)
    print("Haar PSNR hard:", denoiser.calculate_psnr(original, hard_rec))
    print("Haar PSNR soft:", denoiser.calculate_psnr(original, soft_rec))
    Image.fromarray(hard_rec.astype(np.uint8)).save(os.path.join(outdir, "Haar_auto_hard.png"))
    Image.fromarray(soft_rec.astype(np.uint8)).save(os.path.join(outdir, "Haar_auto_soft.png"))

    
    print("Computing DB2 transform...")
    db2_orig = denoiser.multilevel_db2_transform(original, levels)
    db2_noisy = denoiser.multilevel_db2_transform(noisy, levels)
    denoiser.visualize_coefficients(db2_orig, levels, "DB2_Original_Coefficients", outdir)
    denoiser.visualize_coefficients(db2_noisy, levels, "DB2_Noisy_Coefficients", outdir)

    
    t_db2 = denoiser.universal_threshold(db2_noisy, levels)
    print(f"DB2 automatic universal threshold: {t_db2:.3f}")

    hard_db2 = denoiser.hard_threshold(db2_noisy, t_db2, levels)
    soft_db2 = denoiser.soft_threshold(db2_noisy, t_db2, levels)
    hard_db2_rec = np.clip(denoiser.multilevel_db2_inverse(hard_db2, levels), 0, 255)
    soft_db2_rec = np.clip(denoiser.multilevel_db2_inverse(soft_db2, levels), 0, 255)
    print("DB2 PSNR hard:", denoiser.calculate_psnr(original, hard_db2_rec))
    print("DB2 PSNR soft:", denoiser.calculate_psnr(original, soft_db2_rec))
    Image.fromarray(hard_db2_rec.astype(np.uint8)).save(os.path.join(outdir, "DB2_auto_hard.png"))
    Image.fromarray(soft_db2_rec.astype(np.uint8)).save(os.path.join(outdir, "DB2_auto_soft.png"))

    print("Done. Check ./results for outputs.")

if __name__ == "__main__":
    main()
