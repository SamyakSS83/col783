import numpy as np
from PIL import Image
import sys
import os

def gauss_kernel():
    return np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16

def convolve_sep(img, k):
    k = np.asarray(k, dtype=np.float32)
    pad = (k.size - 1) // 2
    
    if img.ndim == 2:
        h, w = img.shape
        p = np.pad(img, ((0, 0), (pad, pad)), mode='reflect')
        temp = np.zeros((h, w), dtype=np.float32)
        for i, kv in enumerate(k):
            temp += kv * p[:, i:i + w]
        pv = np.pad(temp, ((pad, pad), (0, 0)), mode='reflect')
        res = np.zeros((h, w), dtype=np.float32)
        for i, kv in enumerate(k):
            res += kv * pv[i:i + h, :]
        return res.astype(img.dtype)
    else:
        h, w, ch = img.shape
        p = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode='reflect')
        temp = np.zeros((h, w, ch), dtype=np.float32)
        for i, kv in enumerate(k):
            temp += kv * p[:, i:i + w, :]
        pv = np.pad(temp, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
        res = np.zeros((h, w, ch), dtype=np.float32)
        for i, kv in enumerate(k):
            res += kv * pv[i:i + h, :, :]
        return res.astype(img.dtype)

def downsample(img):
    return img[::2, ::2]

def upsample(img, shape):
    h, w = shape[:2]
    temp = np.zeros((h, w, img.shape[2]) if len(img.shape) == 3 else (h, w), dtype=img.dtype)
    temp[::2, ::2] = img
    return temp

def gauss_pyramid(img, levels):
    k = gauss_kernel()
    pyr = [img]
    for i in range(levels):
        img = convolve_sep(img, k)
        img = downsample(img)
        pyr.append(img)
    return pyr

def laplacian_pyramid(gauss_pyr):
    k = gauss_kernel()
    lap_pyr = []
    for i in range(len(gauss_pyr) - 1):
        up = upsample(gauss_pyr[i + 1], gauss_pyr[i].shape)
        up = convolve_sep(up, k) * 4
        lap = gauss_pyr[i] - up
        lap_pyr.append(lap)
    lap_pyr.append(gauss_pyr[-1])
    return lap_pyr

def reconstruct(lap_pyr):
    k = gauss_kernel()
    img = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        img = upsample(img, lap_pyr[i].shape)
        img = convolve_sep(img, k) * 4
        img = img + lap_pyr[i]
    return img

def blend_pyramids(lap1, lap2, mask_pyr):
    blended = []
    for i in range(len(lap1)):
        if len(lap1[i].shape) == 3:
            m = mask_pyr[i][:, :, np.newaxis]
        else:
            m = mask_pyr[i]
        b = (1 - m) * lap1[i] + m * lap2[i]
        blended.append(b)
    return blended

def save_pyramid(pyr, outdir, name):
    for i, img in enumerate(pyr):
        if img.dtype == np.float32 or img.dtype == np.float64:
            temp = img.copy()
            temp = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8) * 255
            temp = temp.astype(np.uint8)
        else:
            temp = img
        
        if temp.ndim == 3 and temp.shape[2] == 3:
            Image.fromarray(temp.astype(np.uint8)).save(os.path.join(outdir, f'{name}_level_{i}.png'))
        else:
            Image.fromarray(temp.astype(np.uint8)).save(os.path.join(outdir, f'{name}_level_{i}.png'))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python3 blend.py <img_left> <img_right> <outdir>')
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    outdir = sys.argv[3]
    
    os.makedirs(outdir, exist_ok=True)
    
    f = np.array(Image.open(img1_path).convert('RGB'), dtype=np.float32)
    g = np.array(Image.open(img2_path).convert('RGB'), dtype=np.float32)
    
    if f.shape != g.shape:
        print('Error: Images must be same size')
        sys.exit(1)
    
    h, w = f.shape[:2]
    
    
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:, w//2:] = 1.0
    
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(outdir, 'mask.png'))
    
    levels = 6
    
    print('Building Gaussian pyramid for image 1...')
    gp_f = gauss_pyramid(f, levels)
    save_pyramid(gp_f, outdir, 'gauss_f')
    
    print('Building Gaussian pyramid for image 2...')
    gp_g = gauss_pyramid(g, levels)
    save_pyramid(gp_g, outdir, 'gauss_g')
    
    print('Building Laplacian pyramid for image 1...')
    lp_f = laplacian_pyramid(gp_f)
    save_pyramid(lp_f, outdir, 'laplacian_f')
    
    print('Building Laplacian pyramid for image 2...')
    lp_g = laplacian_pyramid(gp_g)
    save_pyramid(lp_g, outdir, 'laplacian_g')
    
    print('Reconstructing image 1 from Laplacian pyramid...')
    recon = reconstruct(lp_f)
    recon = np.clip(recon, 0, 255).astype(np.uint8)
    Image.fromarray(recon).save(os.path.join(outdir, 'reconstructed_f.png'))
    
    print('Creating hard blend...')
    hard = (1 - mask[:, :, np.newaxis]) * f + mask[:, :, np.newaxis] * g
    hard = np.clip(hard, 0, 255).astype(np.uint8)
    Image.fromarray(hard).save(os.path.join(outdir, 'hard_blend.png'))
    
    print('Building Gaussian pyramid for mask...')
    gp_mask = gauss_pyramid(mask, levels)
    save_pyramid(gp_mask, outdir, 'gauss_mask')
    
    print('Blending Laplacian pyramids...')
    blended_lap = blend_pyramids(lp_f, lp_g, gp_mask)
    save_pyramid(blended_lap, outdir, 'blended_laplacian')
    
    print('Reconstructing final smooth blend...')
    final = reconstruct(blended_lap)
    final = np.clip(final, 0, 255).astype(np.uint8)
    Image.fromarray(final).save(os.path.join(outdir, 'final_smooth_blend.png'))
    
    print(f'Done! Check {outdir}/ for results')