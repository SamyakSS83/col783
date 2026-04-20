import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, fft
from PIL import Image, ImageDraw
import time


def load_and_prepare_psf(filename, target_sizes):
    img = Image.open(filename).convert('L')
    original_size = img.size[0]
    print(f"  Original PSF size: {original_size}×{original_size}")
    psf_original = np.array(img, dtype=np.float64)
    
    if psf_original.mean() > 128:
        psf_original = 255 - psf_original
        print("Inverted PSF (was dark on bright background)")
    
    psf_original = psf_original / psf_original.sum()

    psfs = {}
    for size in target_sizes:
        if size == original_size:
            psfs[size] = psf_original
        else:
            # Resize using PIL
            psf_img = Image.fromarray((psf_original * 255 / psf_original.max()).astype(np.uint8))
            psf_img = psf_img.resize((size, size), Image.LANCZOS)
            psf_resized = np.array(psf_img, dtype=np.float64)
            psf_resized = psf_resized / psf_resized.sum()
            psfs[size] = psf_resized
        
        psf_img = (psfs[size] * 255 / psfs[size].max()).astype(np.uint8)
        Image.fromarray(psf_img).save(f'psf_{size}.png')
        print(f"  Created psf_{size}.png")
    
    return psfs

def blur_spatial(image, kernel, save_padded=False, filename_prefix=''):
    if save_padded:
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        padded_img_normalized = (padded_image * 255).astype(np.uint8)
        Image.fromarray(padded_img_normalized).save(f'{filename_prefix}_mirror_padded.png')
        print(f"  Saved mirror-padded image: {filename_prefix}_mirror_padded.png")
    
    result = ndimage.convolve(image, kernel, mode='reflect')
    return result

def blur_frequency(image, kernel):
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    padded_kernel = np.zeros_like(image)
    
    start_h = (img_h - ker_h) // 2
    start_w = (img_w - ker_w) // 2
    padded_kernel[start_h:start_h+ker_h, start_w:start_w+ker_w] = kernel

    padded_kernel /= padded_kernel.sum()
    
    padded_kernel = np.fft.ifftshift(padded_kernel)
    img_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(padded_kernel)
    result_fft = img_fft * kernel_fft
    result = np.fft.ifft2(result_fft).real
    
    img_fft_mag = np.log(1 + np.abs(np.fft.fftshift(img_fft)))
    kernel_fft_mag = np.log(1 + np.abs(np.fft.fftshift(kernel_fft)))
    result_fft_mag = np.log(1 + np.abs(np.fft.fftshift(result_fft)))
    
    return result, img_fft_mag, kernel_fft_mag, result_fft_mag

def create_impulse_image(size=1000):
    img = np.zeros((size, size))
    positions = [(250, 250), (250, 750), (500, 500), (750, 250), (750, 750)]
    for y, x in positions:
        img[y, x] = 1.0
    return img

def load_photograph(filename, target_size=(1000, 2000)):
    try:
        img = Image.open(filename).convert('L')
        print(f"  Loaded {filename} - Original size: {img.size}")
        img = img.resize(target_size, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float64) / 255.0
        return img_array
    except FileNotFoundError:
        print(f"  ERROR: Could not find {filename}")
        print("  Please provide a photograph as described in the suggestions.")
        raise

def apply_gamma(image, gamma=2.2):
    return np.power(np.clip(image, 0, 1), gamma)

def undo_gamma(image, gamma=2.2):
    return np.power(np.clip(image, 0, 1), 1/gamma)

def demonstrate_impulse_blurring(impulse_img, psf_small):
    t_start = time.time()
    # blurred_spatial = blur_spatial(impulse_img, psf_small)
    blurred_spatial = blur_spatial(impulse_img, psf_small, save_padded=True, filename_prefix='impulse')
    t_spatial = time.time() - t_start
    print(f"Spatial domain time: {t_spatial:.4f} seconds")
    
    t_start = time.time()
    blurred_freq, img_fft, ker_fft, res_fft = blur_frequency(impulse_img, psf_small)
    t_freq = time.time() - t_start
    print(f"Frequency domain time: {t_freq:.4f} seconds")
    
    # Check if results match
    max_diff = np.max(np.abs(blurred_spatial - blurred_freq))
    print(f"Maximum difference between methods: {max_diff:.2e}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(impulse_img, cmap='gray')
    axes[0, 0].set_title('Original Impulse Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred_spatial, cmap='gray')
    axes[0, 1].set_title(f'Spatial Domain Blur\n({t_spatial:.4f}s)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(blurred_freq, cmap='gray')
    axes[0, 2].set_title(f'Frequency Domain Blur\n({t_freq:.4f}s)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_fft, cmap='gray')
    axes[1, 0].set_title('Image FFT (log magnitude)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ker_fft, cmap='gray')
    axes[1, 1].set_title('Kernel FFT (log magnitude)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(res_fft, cmap='gray')
    axes[1, 2].set_title('Result FFT (log magnitude)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('blur_comparison_impulse.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: blur_comparison_impulse.png")
    
    return t_spatial, t_freq

def demonstrate_photo_blurring(photo_img, psf_small, use_gamma=True):
    if use_gamma:
        print("Applying gamma transformation (γ=2.2) for better visual results...")
        photo_processed = apply_gamma(photo_img, 2.2)
    else:
        photo_processed = photo_img
    
    t_start = time.time()
    # blurred_spatial = blur_spatial(photo_processed, psf_small)
    blurred_spatial = blur_spatial(photo_processed, psf_small, save_padded=True, filename_prefix='photo')    
    t_spatial = time.time() - t_start
    print(f"Spatial domain time: {t_spatial:.4f} seconds")
    
    t_start = time.time()
    blurred_freq, img_fft, ker_fft, res_fft = blur_frequency(photo_processed, psf_small)
    t_freq = time.time() - t_start
    print(f"Frequency domain time: {t_freq:.4f} seconds")
    
    if use_gamma:
        print("Undoing gamma transformation...")
        blurred_spatial = undo_gamma(blurred_spatial, 2.2)
        blurred_freq = undo_gamma(blurred_freq, 2.2)
    
    max_diff = np.max(np.abs(blurred_spatial - blurred_freq))
    print(f"Maximum difference between methods: {max_diff:.2e}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(photo_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Photograph')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred_spatial, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Spatial Domain Blur\n({t_spatial:.4f}s)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(blurred_freq, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Frequency Domain Blur\n({t_freq:.4f}s)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_fft, cmap='gray')
    axes[1, 0].set_title('Image FFT (log magnitude)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ker_fft, cmap='gray')
    axes[1, 1].set_title('Kernel FFT (log magnitude)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(res_fft, cmap='gray')
    axes[1, 2].set_title('Result FFT (log magnitude)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('blur_comparison_photo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: blur_comparison_photo.png")
    
    return t_spatial, t_freq

def perform_timing_analysis(test_img, psfs, sizes):
    timing_results = {'sizes': [], 'spatial': [], 'frequency': []}
    
    for size in sizes:
        print(f"\nTesting with PSF size {size}×{size}...")
        psf = psfs[size]
        
        # Spatial domain
        t_start = time.time()
        _ = blur_spatial(test_img, psf)
        t_spatial = time.time() - t_start
        
        # Frequency domain
        t_start = time.time()
        _, _, _, _ = blur_frequency(test_img, psf)
        t_freq = time.time() - t_start
        
        timing_results['sizes'].append(size)
        timing_results['spatial'].append(t_spatial)
        timing_results['frequency'].append(t_freq)
        
        print(f"  Spatial domain: {t_spatial:.4f}s")
        print(f"  Frequency domain: {t_freq:.4f}s")
        print(f"  Speedup: {t_spatial/t_freq:.2f}x")
    
    # Plot timing results on log-log scale
    plt.figure(figsize=(10, 7))
    plt.loglog(timing_results['sizes'], timing_results['spatial'], 
               'o-', linewidth=2, markersize=8, label='Spatial Domain')
    plt.loglog(timing_results['sizes'], timing_results['frequency'], 
               's-', linewidth=2, markersize=8, label='Frequency Domain')
    plt.xlabel('Kernel Size max(m, n)', fontsize=12)
    plt.ylabel('Computation Time T(τ) [seconds]', fontsize=12)
    plt.title('Blurring Performance: Spatial vs Frequency Domain', fontsize=14)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('timing_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: timing_comparison.png")
    
    return timing_results

def main():
    PSF_FILENAME = 'h.png'
    PHOTO_FILENAME = 'p.png'
    PSF_SIZES = [10, 20, 40, 80, 160]
    
    psfs = load_and_prepare_psf(PSF_FILENAME, PSF_SIZES)
    
    psf_small = psfs[PSF_SIZES[0]]
    
    # Test 1: Impulse image
    impulse_img = create_impulse_image(1000)
    demonstrate_impulse_blurring(impulse_img, psf_small)
    
    # Test 2: Photograph
    photo_img = load_photograph(PHOTO_FILENAME, (1000, 2000))
    demonstrate_photo_blurring(photo_img, psf_small, use_gamma=True)
    
    test_img = create_impulse_image(1000)
    timing_results = perform_timing_analysis(test_img, psfs, PSF_SIZES)

if __name__ == "__main__":    
    main()