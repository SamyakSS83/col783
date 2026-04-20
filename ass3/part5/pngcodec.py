#!/usr/bin/env python3
import sys, os, math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from array import array

def paeth(a,b,c):
    a = int(a); b = int(b); c = int(c)
    p = a + b - c
    pa = abs(p - a); pb = abs(p - b); pc = abs(p - c)
    if pa <= pb and pa <= pc: return a
    if pb <= pc: return b
    return c

def pred_none(row, i, j, img):
    if len(img.shape) == 3:
        return np.zeros(3, dtype=np.int32)
    return 0

def pred_left(row, i, j, img):
    if j > 0:
        return row[j-1].astype(np.int32) if len(img.shape) == 3 else int(row[j-1])
    return np.zeros(3, dtype=np.int32) if len(img.shape) == 3 else 0

def pred_up(row, i, j, img):
    if i > 0:
        return img[i-1,j].astype(np.int32) if len(img.shape) == 3 else int(img[i-1,j])
    return np.zeros(3, dtype=np.int32) if len(img.shape) == 3 else 0

def pred_avg(row, i, j, img):
    if len(img.shape) == 3:
        a = row[j-1].astype(np.int32) if j>0 else np.zeros(3, dtype=np.int32)
        b = img[i-1,j].astype(np.int32) if i>0 else np.zeros(3, dtype=np.int32)
        return ((a + b) // 2).astype(np.int32)
    else:
        a = int(row[j-1]) if j>0 else 0
        b = int(img[i-1,j]) if i>0 else 0
        return (a + b) // 2

def pred_paeth_fn(row, i, j, img):
    if len(img.shape) == 3:
        if i==0 and j==0: 
            return np.zeros(3, dtype=np.int32)
        a = row[j-1].astype(np.int32) if j>0 else np.zeros(3, dtype=np.int32)
        b = img[i-1,j].astype(np.int32) if i>0 else np.zeros(3, dtype=np.int32)
        c = img[i-1,j-1].astype(np.int32) if i>0 and j>0 else np.zeros(3, dtype=np.int32)
        return np.array([paeth(a[ch], b[ch], c[ch]) for ch in range(3)], dtype=np.int32)
    else:
        if i==0 and j==0: return 0
        a = int(row[j-1]) if j>0 else 0
        b = int(img[i-1,j]) if i>0 else 0
        c = int(img[i-1,j-1]) if i>0 and j>0 else 0
        return paeth(a, b, c)



def predict_row(img):
    
    if len(img.shape) == 3:
        h, w, ch = img.shape
        e = np.zeros_like(img, dtype=np.int16)
        
        for i in range(h):
            for j in range(w):
                if i==0 and j==0:
                    pred = np.zeros(ch, dtype=np.int32)
                elif i==0:
                    pred = img[i,j-1].astype(np.int32)
                elif j==0:
                    pred = img[i-1,j].astype(np.int32)
                else:
                    a = img[i,j-1].astype(np.int32)
                    b = img[i-1,j].astype(np.int32)
                    c = img[i-1,j-1].astype(np.int32)
                    pred = np.array([paeth(a[k], b[k], c[k]) for k in range(ch)], dtype=np.int32)
                e[i,j] = img[i,j].astype(np.int32) - pred
        return e
    else:
        h,w = img.shape
        e = np.zeros_like(img, dtype=np.int16)
        for i in range(h):
            for j in range(w):
                if i==0 and j==0:
                    pred = 0
                elif i==0:
                    pred = int(img[i,j-1])
                elif j==0:
                    pred = int(img[i-1,j])
                else:
                    pred = paeth(int(img[i,j-1]), int(img[i-1,j]), int(img[i-1,j-1]))
                e[i,j] = int(img[i,j]) - pred
        return e

def predict_row_perrow(img):
    
    preds = [pred_none, pred_left, pred_up, pred_avg, pred_paeth_fn]
    
    if len(img.shape) == 3:
        h, w, ch = img.shape
        e = np.zeros_like(img, dtype=np.int16)
        modes = []
        
        for i in range(h):
            best_mode = 0
            best_err = None
            
            for mi, pfn in enumerate(preds):
                row = np.zeros((w, ch), dtype=np.uint8)
                err = 0
                for j in range(w):
                    p = pfn(row, i, j, img)
                    row[j] = img[i,j]
                    diff = img[i,j].astype(np.int32) - p
                    err += np.sum(np.abs(diff))
                
                if best_err is None or err < best_err:
                    best_err = err
                    best_mode = mi
            
            modes.append(best_mode)
            pfn = preds[best_mode]
            row = np.zeros((w, ch), dtype=np.uint8)
            
            for j in range(w):
                p = pfn(row, i, j, img)
                row[j] = img[i,j]
                e[i,j] = img[i,j].astype(np.int32) - p
        
        return e, modes
    else:
        h,w = img.shape
        e = np.zeros_like(img, dtype=np.int16)
        modes = []
        
        for i in range(h):
            best_mode = 0
            best_err = None
            for mi, pfn in enumerate(preds):
                row = np.zeros(w, dtype=np.uint8)
                err = 0
                for j in range(w):
                    p = pfn(row, i, j, img)
                    row[j] = img[i,j]
                    diff = int(img[i,j]) - int(p)
                    err += abs(diff)
                if best_err is None or err < best_err:
                    best_err = err
                    best_mode = mi
            
            modes.append(best_mode)
            pfn = preds[best_mode]
            row = np.zeros(w, dtype=np.uint8)
            for j in range(w):
                p = pfn(row, i, j, img)
                row[j] = img[i,j]
                e[i,j] = int(img[i,j]) - int(p)
        
        return e, modes



def recover_from_error(e, modes=None, img_shape=None):
    
    preds = [pred_none, pred_left, pred_up, pred_avg, pred_paeth_fn]
    
    if len(e.shape) == 3:
        h, w, ch = e.shape
        f = np.zeros_like(e, dtype=np.uint8)
        
        for i in range(h):
            if modes is None:
                pfn = pred_paeth_fn
            else:
                pfn = preds[modes[i]]
            
            for j in range(w):
                if modes is None:
                    if i==0 and j==0:
                        pred = np.zeros(ch, dtype=np.int32)
                    elif i==0:
                        pred = f[i,j-1].astype(np.int32)
                    elif j==0:
                        pred = f[i-1,j].astype(np.int32)
                    else:
                        a = f[i,j-1].astype(np.int32)
                        b = f[i-1,j].astype(np.int32)
                        c = f[i-1,j-1].astype(np.int32)
                        pred = np.array([paeth(a[k], b[k], c[k]) for k in range(ch)], dtype=np.int32)
                else:
                    pred = pfn(f[i], i, j, f)
                
                val = pred + e[i,j].astype(np.int32)
                f[i,j] = np.clip(val, 0, 255).astype(np.uint8)
        return f
    else:
        h,w = e.shape
        f = np.zeros_like(e, dtype=np.uint8)
        
        for i in range(h):
            if modes is None:
                pfn = pred_paeth_fn
            else:
                pfn = preds[modes[i]]
            
            for j in range(w):
                if modes is None:
                    if i==0 and j==0:
                        pred = 0
                    elif i==0:
                        pred = int(f[i,j-1])
                    elif j==0:
                        pred = int(f[i-1,j])
                    else:
                        pred = paeth(int(f[i,j-1]), int(f[i-1,j]), int(f[i-1,j-1]))
                else:
                    pred = pfn(f[i], i, j, f)
                
                f[i,j] = np.uint8((int(pred) + int(e[i,j])) & 0xFF)
        return f


def hist_stats(e):
    
    flat = e.flatten()
    
    flat_mapped = (flat + 255).astype(np.int32)
    cnt = Counter(flat_mapped)
    
    total = len(flat)
    probs = []
    for i in range(512):
        p = cnt.get(i, 0) / total
        probs.append(p)
    
    H = -sum(p * math.log2(p) for p in probs if p > 0)
    std = np.std(flat.astype(np.float64))
    
    return cnt, std, H

def lzw_encode(lst):
    
    
    lst = [(int(x) + 255) % 512 for x in lst]
    
    dict_size = 512
    
    dic = {bytes([i % 256, i // 256]): i for i in range(512)}
    w = b''
    out = []
    
    for val in lst:
        wk = w + bytes([val % 256, val // 256])
        if wk in dic:
            w = wk
        else:
            if w:
                out.append(dic[w])
            if dict_size < 65536:
                dic[wk] = dict_size
                dict_size += 1
            w = bytes([val % 256, val // 256])
    
    if w:
        out.append(dic[w])
    
    return out

def lzw_decode(codes):
    
    if not codes:
        return []
    
    dict_size = 512
    
    dic = {i: bytes([i % 256, i // 256]) for i in range(512)}
    
    w = dic[codes[0]]
    out = bytearray(w)
    
    for k in codes[1:]:
        if k in dic:
            entry = dic[k]
        elif k == dict_size:
            
            entry = w + w[:2]
        else:
            raise ValueError(f'Bad LZW code: {k}')
        
        out.extend(entry)
        
        if dict_size < 65536:
            dic[dict_size] = w + entry[:2]
            dict_size += 1
        
        w = entry
    
    
    result = []
    for i in range(0, len(out), 2):
        if i+1 < len(out):
            val = out[i] + out[i+1] * 256
        else:
            val = out[i]
        result.append(int(val) - 255)
    
    return result

class Node:
    def __init__(self,s,p,left=None,right=None):
        self.s=s; self.p=p; self.left=left; self.right=right

def huffman_build(symp):
    import heapq
    heap = [(p, i, Node(s,p)) for i,(s,p) in enumerate(symp.items())]
    heapq.heapify(heap)
    cnt = len(heap)
    while len(heap)>1:
        p1,_,a = heapq.heappop(heap)
        p2,_,b = heapq.heappop(heap)
        heapq.heappush(heap, (p1+p2, cnt, Node(None, p1+p2, a, b)))
        cnt += 1
    root = heap[0][2]
    table = {}
    def walk(n, code):
        if n.s is not None:
            table[n.s]=tuple(code)
            return
        walk(n.left, code+[0]); walk(n.right, code+[1])
    walk(root, [])
    return table

def huffman_encode(codes, table):
    return [b for c in codes for b in table[c]]

def huffman_decode(bits, table):
    rev = {tuple(v):k for k,v in table.items()}
    out = []
    cur = []
    for b in bits:
        cur.append(b)
        t = tuple(cur)
        if t in rev:
            out.append(rev[t]); cur=[]
    return out

def rle_encode(lst):
    out=[]
    i=0
    n=len(lst)
    while i<n:
        j=i+1
        while j<n and lst[j]==lst[i] and j-i<255:
            j+=1
        out.append(lst[i]); out.append(j-i)
        i=j
    return out

def rle_decode(pairs):
    out=[]
    for i in range(0,len(pairs),2):
        v = pairs[i]; c = pairs[i+1]
        out.extend([v]*c)
    return out


def encode_image(img, use_perrow=False, use_rle=False):
    
    if use_perrow:
        e, modes = predict_row_perrow(img)
    else:
        e = predict_row(img)
        modes = None
    
    flat = list(e.flatten())
    
    if use_rle:
        codes = rle_encode(flat)
    else:
        codes = lzw_encode(flat)
    
    freq = Counter(codes)
    probs = {s: freq[s]/len(codes) for s in freq}
    table = huffman_build(probs)
    bits = huffman_encode(codes, table)
    
    return table, bits, modes, e

def decode_image(shape, table, bits, modes=None, use_rle=False):
    
    codes = huffman_decode(bits, table)
    
    if use_rle:
        flat = rle_decode(codes)
    else:
        flat = lzw_decode(codes)
    
    if len(shape) == 3:
        h, w, ch = shape
        e = np.array(flat[:h*w*ch], dtype=np.int16).reshape(shape)
    else:
        h, w = shape
        e = np.array(flat[:h*w], dtype=np.int16).reshape(shape)
    
    f = recover_from_error(e, modes, shape)
    return f

def save_error_image(e, path):
    
    if len(e.shape) == 3:
        
        vis = (e + 128).astype(np.uint8)
        Image.fromarray(vis).save(path)
    else:
        
        vis = (e + 128).astype(np.uint8)
        Image.fromarray(vis).save(path)

def plot_error_histogram(e, name, outdir):
    
    flat = e.flatten()
    
    plt.figure(figsize=(10, 6))
    plt.hist(flat, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.title(f'{name} Error Histogram')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, f'{name}_hist.png'), dpi=150, bbox_inches='tight')
    plt.close()

def test_lzw_zeros(N):
    
    lst = [0]*N
    codes = lzw_encode(lst)
    dec = lzw_decode(codes)
    ok = dec[:N] == lst
    print(f'  LZW N={N} zeros: ok={ok} out_len={len(codes)}')
    return len(codes)

def run_pipeline(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    
    print('\n' + '='*70)
    print('PART 5: PNG-LIKE COMPRESSION (COLOR)')
    print('='*70)
    
    
    print('\n=== Part (c): LZW on constant sequence ===')
    results_c = []
    for N in [100, 1000, 10000, 100000]:
        out_len = test_lzw_zeros(N)
        results_c.append((N, out_len))
    
    print('\nBig-O Analysis:')
    print('\nEmpirical verification:')
    for N, out_len in results_c:
        ratio = N / (out_len ** 2)
        print(f'  N={N:6d}, len={out_len:4d}, N/len^2={ratio:.2f}')

    imgs = ['f1', 'f2']

    for img_name in imgs:
        print('\n' + '='*70)
        print(f'Processing: {img_name}.png')
        print('='*70)

        path = os.path.join(inp, img_name + '.png')
        im = Image.open(path).convert('RGB')
        a = np.array(im, dtype=np.uint8)
        
        if len(a.shape) == 2:
            print('WARNING: Image is grayscale, converting to RGB')
            a = np.stack([a, a, a], axis=-1)
        
        shape = a.shape
        h, w, ch = shape
        total_pixels = h * w
        
        print(f'Image shape: {h}x{w}x{ch}')
        
        
        Image.fromarray(a).save(os.path.join(outdir, f'{img_name}_original.png'))
        
        
        print('\n--- Part (a): Predictive Coding ---')
        e = predict_row(a)
        
        save_error_image(e, os.path.join(outdir, f'{img_name}_error.png'))
        
        
        for c in range(ch):
            e_ch = e[:,:,c]
            save_error_image(e_ch, os.path.join(outdir, f'{img_name}_error_ch{c}.png'))
        
        cnt, std, H_err = hist_stats(e)
        print(f'Error std dev: {std:.4f}')
        print(f'Error entropy: {H_err:.4f} bits/pixel')
        
        plot_error_histogram(e, img_name, outdir)
        
        
        print('\n--- Part (b): Decoder Verification ---')
        rec = recover_from_error(e)
        Image.fromarray(rec).save(os.path.join(outdir, f'{img_name}_reconstructed.png'))
        
        ok = np.array_equal(rec, a)
        max_diff = np.max(np.abs(rec.astype(np.int32) - a.astype(np.int32)))
        print(f'Perfect reconstruction: {ok}')
        print(f'Max pixel difference: {max_diff}')
        
        
        print('\n--- Part (c) & (d): LZW + Huffman Coding ---')
        flat = list(e.flatten())
        
        print(f'Encoding {len(flat)} error values...')
        codes = lzw_encode(flat)
        print(f'LZW output: {len(codes)} codewords')
        
        
        dec = lzw_decode(codes)
        lzw_ok = dec[:len(flat)] == flat
        print(f'LZW roundtrip OK: {lzw_ok}')
        
        
        freq = Counter(codes)
        probs = {s: freq[s]/len(codes) for s in freq}
        H_lzw = -sum(p * math.log2(p) for p in probs.values() if p > 0)
        
        table = huffman_build(probs)
        bits = huffman_encode(codes, table)
        
        avg_len = sum(len(table[s]) * probs[s] for s in probs)
        
        print(f'LZW codeword entropy: {H_lzw:.4f} bits/symbol')
        print(f'Huffman avg length: {avg_len:.4f} bits/symbol')
        print(f'Huffman efficiency: {H_lzw/avg_len:.6f}')
        print(f'Total encoded bits: {len(bits)}')
        
        
        print('\n--- Part (e): Compression Ratios ---')
        
        original_bits = h * w * ch * 8
        ratio_pred = 8.0 / H_err if H_err > 0 else float('inf')
        ratio_lzw = len(flat) / len(codes)
        ratio_total = original_bits / len(bits)
        
        print(f'Expected from prediction: 8/{H_err:.4f} = {ratio_pred:.4f}x')
        print(f'LZW compression: {len(flat)}/{len(codes)} = {ratio_lzw:.4f}x')
        print(f'Overall compression: {original_bits}/{len(bits)} = {ratio_total:.4f}x')
        print(f'Final size: {len(bits)/8:.0f} bytes (from {original_bits/8:.0f} bytes)')
        
        
        print('\n--- Part (e): Full Pipeline Test ---')
        T, B, _, e_test = encode_image(a)
        rec_full = decode_image(shape, T, B)
        Image.fromarray(rec_full).save(os.path.join(outdir, f'{img_name}_fullpipeline_reconstructed.png'))
        
        pipeline_ok = np.array_equal(rec_full, a)
        print(f'Full pipeline reconstruction: {pipeline_ok}')
        
        
        with open(os.path.join(outdir, f'{img_name}_stats.txt'), 'w') as f:
            f.write(f'Image: {img_name}\n')
            f.write(f'Size: {h}x{w}x{ch}\n')
            f.write(f'Error std dev: {std:.4f}\n')
            f.write(f'Error entropy: {H_err:.4f} bits/pixel\n')
            f.write(f'LZW codewords: {len(codes)}\n')
            f.write(f'LZW entropy: {H_lzw:.4f} bits/symbol\n')
            f.write(f'Huffman avg length: {avg_len:.4f} bits/symbol\n')
            f.write(f'Prediction ratio: {ratio_pred:.4f}x\n')
            f.write(f'LZW ratio: {ratio_lzw:.4f}x\n')
            f.write(f'Overall ratio: {ratio_total:.4f}x\n')
            f.write(f'Original size: {original_bits/8:.0f} bytes\n')
            f.write(f'Compressed size: {len(bits)/8:.0f} bytes\n')
    
    
    print('\n' + '='*70)
    print('PART (f): Per-row Predictor + RLE')
    print('='*70)
    
    outdir_bonus = outdir + '_bonus'
    os.makedirs(outdir_bonus, exist_ok=True)
    
    for img_name in imgs:
        print(f'\nProcessing {img_name}.png with per-row predictor selection...')

        path = os.path.join(inp, img_name + '.png')
        im = Image.open(path).convert('RGB')
        a = np.array(im, dtype=np.uint8)
        
        if len(a.shape) == 2:
            a = np.stack([a, a, a], axis=-1)
        
        shape = a.shape
        h, w, ch = shape
        
        
        e_perrow, modes = predict_row_perrow(a)
        
        save_error_image(e_perrow, os.path.join(outdir_bonus, f'{img_name}_error_perrow.png'))
        
        cnt_perrow, std_perrow, H_perrow = hist_stats(e_perrow)
        plot_error_histogram(e_perrow, f'{img_name}_perrow', outdir_bonus)
        
        print(f'  Per-row error entropy: {H_perrow:.4f} bits/pixel')
        
        
        with open(os.path.join(outdir_bonus, f'{img_name}_modes.txt'), 'w') as f:
            f.write(','.join(map(str, modes)))
        
        mode_names = ['none', 'left', 'up', 'avg', 'paeth']
        mode_counts = Counter(modes)
        print('  Predictor usage:')
        for mode_idx, count in sorted(mode_counts.items()):
            print(f'    {mode_names[mode_idx]}: {count} rows ({100*count/len(modes):.1f}%)')
        
        
        T_perrow, B_perrow, modes_enc, _ = encode_image(a, use_perrow=True, use_rle=False)
        ratio_perrow_lzw = (h*w*ch*8) / len(B_perrow)
        print(f'  Per-row + LZW compression: {ratio_perrow_lzw:.4f}x')
        
        
        T_rle, B_rle, modes_rle, _ = encode_image(a, use_perrow=True, use_rle=True)
        ratio_perrow_rle = (h*w*ch*8) / len(B_rle)
        print(f'  Per-row + RLE compression: {ratio_perrow_rle:.4f}x')
        
        
        rec_perrow = decode_image(shape, T_perrow, B_perrow, modes_enc, use_rle=False)
        Image.fromarray(rec_perrow).save(os.path.join(outdir_bonus, f'{img_name}_reconstructed_perrow.png'))
        ok_perrow = np.array_equal(rec_perrow, a)
        print(f'  Per-row+LZW reconstruction OK: {ok_perrow}')
        
        rec_rle = decode_image(shape, T_rle, B_rle, modes_rle, use_rle=True)
        Image.fromarray(rec_rle).save(os.path.join(outdir_bonus, f'{img_name}_reconstructed_rle.png'))
        ok_rle = np.array_equal(rec_rle, a)
        print(f'  Per-row+RLE reconstruction OK: {ok_rle}')
        
        
        with open(os.path.join(outdir_bonus, f'{img_name}_bonus_stats.txt'), 'w') as f:
            f.write(f'Image: {img_name}\n')
            f.write(f'Per-row entropy: {H_perrow:.4f} bits/pixel\n')
            f.write(f'Per-row + LZW ratio: {ratio_perrow_lzw:.4f}x\n')
            f.write(f'Per-row + RLE ratio: {ratio_perrow_rle:.4f}x\n')
            f.write(f'\nPredictor usage:\n')
            for mode_idx, count in sorted(mode_counts.items()):
                f.write(f'  {mode_names[mode_idx]}: {count} rows ({100*count/len(modes):.1f}%)\n')
    
    print('\n' + '='*70)
    print('COMPRESSION ANALYSIS COMPLETE')
    print('='*70)
    print(f'\nAll results saved to:')
    print(f'  Main: {outdir}/')
    print(f'  Bonus: {outdir_bonus}/')

if __name__=='__main__':
    if len(sys.argv)<3:
        print('usage: python3 pngcodec.py <indir> <outdir>')
        sys.exit(1)
    run_pipeline(sys.argv[1], sys.argv[2])
