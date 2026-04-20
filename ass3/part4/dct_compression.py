import sys, os
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def block_dct_forward(img, bs=16):
    h, w = img.shape
    coeffs = np.zeros_like(img, dtype=np.float64)
    for i in range(0, h, bs):
        for j in range(0, w, bs):
            block = img[i:i+bs, j:j+bs].astype(np.float64)
            coeffs[i:i+bs, j:j+bs] = dct2(block)
    return coeffs

def block_dct_inverse(coeffs, bs=16):
    h, w = coeffs.shape
    img = np.zeros_like(coeffs, dtype=np.float64)
    for i in range(0, h, bs):
        for j in range(0, w, bs):
            block = coeffs[i:i+bs, j:j+bs]
            img[i:i+bs, j:j+bs] = idct2(block)
    return img

def full_dct_forward(img):
    return dct2(img.astype(np.float64))

def full_dct_inverse(coeffs):
    return idct2(coeffs)

def threshold_block_by_rmse(coeffs, bs, max_rmse):
    
    
    
    flat = coeffs.flatten()
    idx = np.argsort(np.abs(flat))
    keep = flat.copy()
    N = bs * bs
    threshold_sq = (max_rmse ** 2) * N
    
    dropped_energy = 0
    for i in range(len(idx)):
        candidate = idx[i]
        val = flat[candidate]
        new_energy = dropped_energy + val**2
        if new_energy <= threshold_sq:
            keep[candidate] = 0
            dropped_energy = new_energy
        else:
            break
    
    return keep.reshape(coeffs.shape)

def compress_image_block(img, max_rmse, bs=16):
    h, w = img.shape
    coeffs = block_dct_forward(img, bs)
    compressed = np.zeros_like(coeffs)
    total_nonzero = 0
    
    for i in range(0, h, bs):
        for j in range(0, w, bs):
            block = coeffs[i:i+bs, j:j+bs]
            thresh = threshold_block_by_rmse(block, bs, max_rmse)
            compressed[i:i+bs, j:j+bs] = thresh
            total_nonzero += np.count_nonzero(thresh)
    
    recon = block_dct_inverse(compressed, bs)
    return recon, compressed, total_nonzero

def compress_image_full(img, max_rmse):
    coeffs = full_dct_forward(img)
    h, w = img.shape
    N = h * w
    threshold_sq = (max_rmse ** 2) * N
    
    flat = coeffs.flatten()
    idx = np.argsort(np.abs(flat))
    keep = flat.copy()
    
    dropped_energy = 0
    for i in range(len(idx)):
        candidate = idx[i]
        val = flat[candidate]
        new_energy = dropped_energy + val**2
        if new_energy <= threshold_sq:
            keep[candidate] = 0
            dropped_energy = new_energy
        else:
            break
    
    compressed = keep.reshape(coeffs.shape)
    recon = full_dct_inverse(compressed)
    total_nonzero = np.count_nonzero(compressed)
    
    return recon, compressed, total_nonzero

def psnr(orig, recon):
    mse = np.mean((orig.astype(np.float64) - recon)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def save_img(arr, path):
    a = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)

def process_image(path, name, outdir, max_rmse=25.5):
    print(f'\n=== Processing {name} ===')
    img = Image.open(path).convert('L')
    a = np.array(img, dtype=np.float64)
    h, w = a.shape    
    h_new = (h // 16) * 16
    w_new = (w // 16) * 16
    if h_new != h or w_new != w:
        a = a[:h_new, :w_new]
    save_img(a, os.path.join(outdir, f'{name}_orig.png'))
    
    
    print('Part (a): Testing DCT forward/inverse')
    coeffs_test = block_dct_forward(a, 16)
    recon_test = block_dct_inverse(coeffs_test, 16)
    max_err = np.max(np.abs(a - recon_test))
    err = np.abs(a - recon_test)
    print(f'Max reconstruction error: {max_err:.2e}')

    save_img(recon_test, os.path.join(outdir, f'{name}_recon_part_a.png'))
    save_img(err, os.path.join(outdir, f'{name}_error_part_a.png'))
    
    
    print(f'Part (b): Block DCT with RMSE <= {max_rmse:.1f}')
    recon_block, coeffs_block, nonzero_block = compress_image_block(a, max_rmse, 16)
    psnr_block = psnr(a, recon_block)
    print(f'Block DCT: PSNR={psnr_block:.2f} dB, nonzero={nonzero_block}/{h_new*w_new}')
    
    save_img(recon_block, os.path.join(outdir, f'{name}_recon_part_b.png'))
    
    err_block = np.abs(a - recon_block)
    save_img(err_block, os.path.join(outdir, f'{name}_block_error_part_b.png'))
    
    
    print(f'Part (c): Full DCT with RMSE <= {max_rmse:.1f}')
    recon_full, coeffs_full, nonzero_full = compress_image_full(a, max_rmse)
    psnr_full = psnr(a, recon_full)
    print(f'Full DCT: PSNR={psnr_full:.2f} dB, nonzero={nonzero_full}/{h_new*w_new}')
    
    save_img(recon_full, os.path.join(outdir, f'{name}_full_recon.png'))
    
    err_full = np.abs(a - recon_full)
    save_img(err_full, os.path.join(outdir, f'{name}_full_error.png'))
    
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0,0].imshow(a, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(recon_block, cmap='gray', vmin=0, vmax=255)
    axes[0,1].set_title(f'Block DCT\nPSNR={psnr_block:.1f}dB')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(err_block, cmap='hot', vmin=0, vmax=50)
    axes[0,2].set_title('Block Error')
    axes[0,2].axis('off')
    
    axes[1,0].axis('off')
    
    axes[1,1].imshow(recon_full, cmap='gray', vmin=0, vmax=255)
    axes[1,1].set_title(f'Full DCT\nPSNR={psnr_full:.1f}dB')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(err_full, cmap='hot', vmin=0, vmax=50)
    axes[1,2].set_title('Full Error')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{name}_comparison.png'), dpi=150)
    plt.close()
    
    return {
        'psnr_block': psnr_block,
        'nonzero_block': nonzero_block,
        'psnr_full': psnr_full,
        'nonzero_full': nonzero_full,
        'total_coeffs': h_new * w_new
    }

def main():
    if len(sys.argv) < 3:
        print('usage: python3 dct_compression.py <img1> <img2> <outdir>')
        sys.exit(1)
    
    img1 = sys.argv[1]
    img2 = sys.argv[2]
    outdir = sys.argv[3]
    
    os.makedirs(outdir, exist_ok=True)
    
    print('='*60)
    print('BLOCK DCT COMPRESSION (JPEG-like)')
    print('='*60)
    
    
    max_rmse = 25.5
    
    r1 = process_image(img1, 'f1', outdir, max_rmse)
    r2 = process_image(img2, 'f2', outdir, max_rmse)
    
    print('\n' + '='*60)
    print('COMPARISON SUMMARY')
    print('='*60)
    print(f'\n{"Metric":<30} {"f1":<15} {"f2":<15}')
    print('-'*60)
    print(f'{"Block DCT PSNR (dB)":<30} {r1["psnr_block"]:<15.2f} {r2["psnr_block"]:<15.2f}')
    print(f'{"Block nonzero coeffs":<30} {r1["nonzero_block"]:<15} {r2["nonzero_block"]:<15}')
    print(f'{"Block retention %":<30} {100*r1["nonzero_block"]/r1["total_coeffs"]:<15.2f} {100*r2["nonzero_block"]/r2["total_coeffs"]:<15.2f}')
    print(f'{"Full DCT PSNR (dB)":<30} {r1["psnr_full"]:<15.2f} {r2["psnr_full"]:<15.2f}')
    print(f'{"Full nonzero coeffs":<30} {r1["nonzero_full"]:<15} {r2["nonzero_full"]:<15}')
    print(f'{"Full retention %":<30} {100*r1["nonzero_full"]/r1["total_coeffs"]:<15.2f} {100*r2["nonzero_full"]/r2["total_coeffs"]:<15.2f}')
    
    print('\n' + '='*60)
    print('OBSERVATIONS')
    print('='*60)
    print('Block DCT:')
    print('  - Localized: errors contained within blocks')
    print('  - Better for images with varying detail density')
    print('  - Standard in JPEG')
    print('\nFull DCT:')
    print('  - Global optimization of coefficient allocation')
    print('  - Can achieve higher PSNR for smooth images')
    print('  - Errors can spread across entire image')
    print('  - Not practical for large images (memory, computation)')
    
    print(f'\nDone! Results saved in {outdir}/')

if __name__ == '__main__':
    main()
