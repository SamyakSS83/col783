import os
import sys
import cv2
import numpy as np


def ensure_dir(path: str) -> str:
    # If path is empty, treat as current directory and do nothing
    if not path or path.strip() == "":
        return os.getcwd()
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def load_image_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: could not load '{path}'")
        sys.exit(1)
    return img


def make_logo_mask_white_bg(logo_bgr: np.ndarray) -> np.ndarray:
    
    # Convert BGR to grayscale via numpy (avoid heavy libraries)
    b = logo_bgr[..., 0].astype(np.float32)
    g = logo_bgr[..., 1].astype(np.float32)
    r = logo_bgr[..., 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r

    # Pick a high threshold near white; guard via percentile
    t = max(220.0, float(np.percentile(gray, 95)))
    mask = (gray < t).astype(np.float32)  # 1 where logo ink exists, 0 on white
    return mask


def resize_logo_and_mask(logo_bgr: np.ndarray, mask: np.ndarray, doc_w: int) -> tuple:
    target_w = max(1, int(0.20 * doc_w))
    h, w = logo_bgr.shape[:2]
    scale = target_w / float(w)
    target_h = max(1, int(round(h * scale)))

    logo_resized = cv2.resize(logo_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    # Resize mask then binarize to keep crisp edges
    mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = (mask_resized >= 0.5).astype(np.float32)
    return logo_resized, mask_resized


def blend_bottom_right(doc_bgr: np.ndarray, logo_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    
    H, W = doc_bgr.shape[:2]
    h, w = logo_bgr.shape[:2]
    y0, x0 = H - h, W - w
    if y0 < 0 or x0 < 0:
        print("Error: logo larger than document after resizing")
        sys.exit(1)

    roi = doc_bgr[y0: y0 + h, x0: x0 + w].astype(np.float32)
    logo_f = logo_bgr.astype(np.float32)

    # Expand mask to 3 channels
    M = np.repeat(mask[..., None], 3, axis=2).astype(np.float32)

    out_roi = roi + 0.5 * M * (logo_f - roi)
    out = doc_bgr.copy()
    out[y0: y0 + h, x0: x0 + w] = np.clip(out_roi, 0, 255).astype(np.uint8)
    return out


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python watermark.py <input_image> [logo_image] [output_image]")
        sys.exit(1)

    in_path = sys.argv[1]
    logo_path = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(os.path.dirname(__file__), "iitlogo-23.jpg")
    out_path = sys.argv[3] if len(sys.argv) >= 4 else os.path.join(os.path.dirname(__file__), "results", "watermarked.jpg")

    doc = load_image_color(in_path)
    logo = load_image_color(logo_path)

    mask = make_logo_mask_white_bg(logo)
    logo_r, mask_r = resize_logo_and_mask(logo, mask, doc.shape[1])

    # Align mask to logo placement (done implicitly via ROI placement)
    out = blend_bottom_right(doc, logo_r, mask_r)

    # Save outputs
    results_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "results"))
    mask_vis = (np.clip(mask_r * 255, 0, 255)).astype(np.uint8)
    cv2.imwrite(os.path.join(results_dir, "logo_mask.jpg"), mask_vis)

    # Ensure directory exists for out_path
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, out)
    print(f"Saved: {out_path}\nSaved mask visualization: {os.path.join(results_dir, 'logo_mask.jpg')}.")


if __name__ == "__main__":
    main()
