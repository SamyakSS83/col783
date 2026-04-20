import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

def ensure_dir(path: str) -> str:
    # Ensure directory exists, create if it doesn't
    if not path or path.strip() == "":
        return os.getcwd()
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path

def load_image_color(path: str) -> np.ndarray:
    # Load image in color format
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def save_image(path: str, img: np.ndarray) -> None:
    # Save image to specified path
    ensure_dir(os.path.dirname(path) or ".")
    ext = os.path.splitext(path)[1].lower()
    if ext == "":
        ext = ".jpg"
        path += ext
    ok, enc = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Failed to encode image for {path}")
    with open(path, "wb") as f:
        f.write(enc.tobytes())
    print(f"Saved: {path}")

def rgb_to_hsi(rgb_img: np.ndarray) -> np.ndarray:
    # Convert RGB image to HSI color space
    # Normalize RGB values to [0, 1]
    rgb_normalized = rgb_img.astype(np.float32) / 255.0
    
    R, G, B = rgb_normalized[:, :, 2], rgb_normalized[:, :, 1], rgb_normalized[:, :, 0]
    
    # Calculate Intensity
    I = (R + G + B) / 3.0
    # Calculate Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.zeros_like(I)
    mask = I > 0
    S[mask] = 1 - (min_rgb[mask] / I[mask])
    # Calculate Hue
    H = np.zeros_like(I)

    denominator = np.sqrt((R - G)**2 + (R - B) * (G - B))
    valid_mask = denominator > 1e-6
    
    numerator = 0.5 * ((R - G) + (R - B))
    theta = np.arccos(np.clip(numerator[valid_mask] / denominator[valid_mask], -1, 1))
    
    H[valid_mask] = theta
    H[valid_mask & (B > G)] = 2 * np.pi - H[valid_mask & (B > G)]
    
    # Convert to degrees and normalize to [0, 360]
    H = np.degrees(H)
    
    # Stack HSI channels
    hsi_img = np.stack([H, S, I], axis=2)
    return hsi_img

def hsi_to_rgb(hsi_img: np.ndarray) -> np.ndarray:
    # Convert HSI image back to RGB color space
    H, S, I = hsi_img[:, :, 0], hsi_img[:, :, 1], hsi_img[:, :, 2]
    H_rad = np.radians(H)
    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)
    # Sector 1: 0 <= H < 120
    sector1 = (H >= 0) & (H < 120)
    if np.any(sector1):
        B[sector1] = I[sector1] * (1 - S[sector1])
        R[sector1] = I[sector1] * (1 + (S[sector1] * np.cos(H_rad[sector1])) / 
                                   np.cos(np.pi/3 - H_rad[sector1]))
        G[sector1] = 3 * I[sector1] - (R[sector1] + B[sector1])
    
    # Sector 2: 120 <= H < 240
    sector2 = (H >= 120) & (H < 240)
    if np.any(sector2):
        H_rad_2 = H_rad[sector2] - 2*np.pi/3
        R[sector2] = I[sector2] * (1 - S[sector2])
        G[sector2] = I[sector2] * (1 + (S[sector2] * np.cos(H_rad_2)) / 
                                   np.cos(np.pi/3 - H_rad_2))
        B[sector2] = 3 * I[sector2] - (R[sector2] + G[sector2])
    
    # Sector 3: 240 <= H < 360
    sector3 = (H >= 240) & (H < 360)
    if np.any(sector3):
        H_rad_3 = H_rad[sector3] - 4*np.pi/3
        G[sector3] = I[sector3] * (1 - S[sector3])
        B[sector3] = I[sector3] * (1 + (S[sector3] * np.cos(H_rad_3)) / 
                                   np.cos(np.pi/3 - H_rad_3))
        R[sector3] = 3 * I[sector3] - (G[sector3] + B[sector3])
    
    # Clip values to [0, 1] and convert to [0, 255]
    R = np.clip(R, 0, 1) * 255
    G = np.clip(G, 0, 1) * 255
    B = np.clip(B, 0, 1) * 255
    
    rgb_img = np.stack([B, G, R], axis=2).astype(np.uint8)
    return rgb_img

def display_hsi_channels(hsi_img: np.ndarray, out_dir: str) -> None:
    # Display H, S, and I channels as intensity images
    H, S, I = hsi_img[:, :, 0], hsi_img[:, :, 1], hsi_img[:, :, 2]
    
    # Normalize each channel to [0, 255] for display
    H_display = (H / 360.0 * 255).astype(np.uint8)
    S_display = (S * 255).astype(np.uint8)
    I_display = (I * 255).astype(np.uint8)
    
    # Save individual channels
    save_image(os.path.join(out_dir, "hue_channel.jpg"), H_display)
    save_image(os.path.join(out_dir, "saturation_channel.jpg"), S_display)
    save_image(os.path.join(out_dir, "intensity_channel.jpg"), I_display)
    
    # Create a combined visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(H_display, cmap='hsv')
    axes[0].set_title('Hue Channel')
    axes[0].axis('off')
    
    axes[1].imshow(S_display, cmap='gray')
    axes[1].set_title('Saturation Channel')
    axes[1].axis('off')
    
    axes[2].imshow(I_display, cmap='gray')
    axes[2].set_title('Intensity Channel')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hsi_channels_combined.png"), dpi=150)
    plt.close()

def pick_seed_pixel(img_path: str) -> Tuple[int, int]:
    # Let user pick a seed pixel by clicking on the image
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise RuntimeError(f"cv2.imread failed for {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title("Click on the object you want to select (seed pixel). Press Enter when done.")
    plt.axis('off')
    
    pts = plt.ginput(n=1, timeout=0)
    plt.close()
    
    if len(pts) != 1:
        raise RuntimeError("You must click exactly 1 point as seed pixel.")
    
    x, y = int(pts[0][0]), int(pts[0][1])
    return x, y

def color_slicing_rgb(img: np.ndarray, seed_x: int, seed_y: int, 
                     cube_size: Tuple[int, int, int]) -> np.ndarray:
    # Perform color slicing in RGB color space using a cuboid
    # Get seed color
    seed_color = img[seed_y, seed_x].astype(np.float32)  # BGR format
    
    # Create mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Define cuboid bounds
    lower_bound = seed_color - np.array(cube_size) / 2
    upper_bound = seed_color + np.array(cube_size) / 2
    
    # Apply bounds
    lower_bound = np.maximum(lower_bound, 0)
    upper_bound = np.minimum(upper_bound, 255)
    
    # Create mask based on cuboid
    for i in range(3):  # B, G, R channels
        channel_mask = (img[:, :, i] >= lower_bound[i]) & (img[:, :, i] <= upper_bound[i])
        if i == 0:
            mask = channel_mask.astype(np.uint8)
        else:
            mask = mask & channel_mask.astype(np.uint8)
    
    return mask

def color_slicing_hsi(hsi_img: np.ndarray, seed_x: int, seed_y: int, 
                     cube_size: Tuple[float, float, float]) -> np.ndarray:
    # Perform color slicing in HSI color space using a cuboid
    # Get seed color in HSI
    seed_color = hsi_img[seed_y, seed_x]  # H, S, I
    
    # Create mask
    mask = np.zeros(hsi_img.shape[:2], dtype=np.uint8)
    
    # Define cuboid bounds
    h_range, s_range, i_range = cube_size
    
    # Hue wrapping consideration
    h_lower = seed_color[0] - h_range / 2
    h_upper = seed_color[0] + h_range / 2
    
    # Handle hue wrapping (circular nature)
    if h_lower < 0:
        h_mask = (hsi_img[:, :, 0] >= (h_lower + 360)) | (hsi_img[:, :, 0] <= h_upper)
    elif h_upper > 360:
        h_mask = (hsi_img[:, :, 0] >= h_lower) | (hsi_img[:, :, 0] <= (h_upper - 360))
    else:
        h_mask = (hsi_img[:, :, 0] >= h_lower) & (hsi_img[:, :, 0] <= h_upper)
    
    # Saturation and Intensity masks
    s_mask = (hsi_img[:, :, 1] >= max(0, seed_color[1] - s_range / 2)) & \
             (hsi_img[:, :, 1] <= min(1, seed_color[1] + s_range / 2))
    
    i_mask = (hsi_img[:, :, 2] >= max(0, seed_color[2] - i_range / 2)) & \
             (hsi_img[:, :, 2] <= min(1, seed_color[2] + i_range / 2))
    
    # Combine masks
    mask = (h_mask & s_mask & i_mask).astype(np.uint8)
    
    return mask

def apply_color_transformation(hsi_img: np.ndarray, mask: np.ndarray, 
                              target_hue: float, transform_type: str = 'additive') -> np.ndarray:
    """Apply color transformation to masked pixels in HSI space"""
    result_hsi = hsi_img.copy()
    
    # Get seed pixel values from the first masked pixel
    mask_indices = np.where(mask == 1)
    if len(mask_indices[0]) == 0:
        print("Warning: No pixels in mask, returning original image")
        return result_hsi
    
    seed_hue = hsi_img[mask_indices[0][0], mask_indices[1][0], 0]
    
    if transform_type == 'additive':
        # T(c) = c + (ct - cs)
        hue_shift = target_hue - seed_hue
        result_hsi[mask == 1, 0] = (result_hsi[mask == 1, 0] + hue_shift) % 360
        
    elif transform_type == 'multiplicative':
        # T(c) = (ct/cs) * c
        if seed_hue != 0:
            hue_ratio = target_hue / seed_hue
            result_hsi[mask == 1, 0] = (result_hsi[mask == 1, 0] * hue_ratio) % 360
        else:
            result_hsi[mask == 1, 0] = target_hue
    
    # Keep saturation and intensity unchanged for realistic results
    # Optionally, you could also transform S and I channels
    
    return result_hsi

def visualize_masks_and_results(original_img: np.ndarray, mask_rgb: np.ndarray, 
                               mask_hsi: np.ndarray, out_dir: str) -> None:
    """Visualize original image, masks, and overlay results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert BGR to RGB for display
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Original image
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # RGB mask
    axes[0, 1].imshow(mask_rgb * 255, cmap='gray')
    axes[0, 1].set_title('RGB Color Slicing Mask')
    axes[0, 1].axis('off')
    
    # HSI mask
    axes[0, 2].imshow(mask_hsi * 255, cmap='gray')
    axes[0, 2].set_title('HSI Color Slicing Mask')
    axes[0, 2].axis('off')
    
    # Overlay masks on original
    overlay_rgb = original_rgb.copy()
    overlay_rgb[mask_rgb == 1] = [255, 0, 0]  # Red overlay
    axes[1, 0].imshow(overlay_rgb)
    axes[1, 0].set_title('RGB Mask Overlay')
    axes[1, 0].axis('off')
    
    overlay_hsi = original_rgb.copy()
    overlay_hsi[mask_hsi == 1] = [0, 255, 0]  # Green overlay
    axes[1, 1].imshow(overlay_hsi)
    axes[1, 1].set_title('HSI Mask Overlay')
    axes[1, 1].axis('off')
    
    # Comparison
    comparison = original_rgb.copy()
    comparison[mask_rgb == 1] = [255, 0, 0]    # Red for RGB mask
    comparison[mask_hsi == 1] = [0, 255, 0]    # Green for HSI mask
    comparison[(mask_rgb == 1) & (mask_hsi == 1)] = [255, 255, 0]  # Yellow for overlap
    axes[1, 2].imshow(comparison)
    axes[1, 2].set_title('Mask Comparison\n(Red: RGB, Green: HSI, Yellow: Both)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mask_comparison.png"), dpi=150)
    plt.close()

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python color_manipulation.py <input_image> [output_dir]")
        sys.exit(1)
    
    in_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) == 3 else os.path.join(os.path.dirname(__file__), "results")
    ensure_dir(out_dir)
    
    img = load_image_color(in_path)
    
    # Convert to HSI & Display HSI channels
    hsi_img = rgb_to_hsi(img)
    display_hsi_channels(hsi_img, out_dir)
    
    # Pick seed pixel
    print("Please select a seed pixel on the object you want to modify...")
    seed_x, seed_y = pick_seed_pixel(in_path)
    print(f"Seed pixel selected at: ({seed_x}, {seed_y})")
    
    # Get seed colors
    seed_rgb = img[seed_y, seed_x]
    seed_hsi = hsi_img[seed_y, seed_x]
    print(f"Seed RGB: {seed_rgb}")
    print(f"Seed HSI: {seed_hsi}")
    
    # Perform color slicing in RGB space
    rgb_cube_size = (40, 40, 40)  # Adjust based on your image
    mask_rgb = color_slicing_rgb(img, seed_x, seed_y, rgb_cube_size)
    print(f"RGB mask selected {np.sum(mask_rgb)} pixels")
    
    # Perform color slicing in HSI space
    print("Performing color slicing in HSI space...")
    hsi_cube_size = (30, 0.3, 0.3)  # (Hue range in degrees, Saturation range, Intensity range)
    mask_hsi = color_slicing_hsi(hsi_img, seed_x, seed_y, hsi_cube_size)
    print(f"HSI mask selected {np.sum(mask_hsi)} pixels")
    
    # Save masks
    save_image(os.path.join(out_dir, "mask_rgb.jpg"), mask_rgb * 255)
    save_image(os.path.join(out_dir, "mask_hsi.jpg"), mask_hsi * 255)
    
    # Visualize masks and results
    visualize_masks_and_results(img, mask_rgb, mask_hsi, out_dir)
    
    # Apply color transformations with different target colors
    target_hues = [60, 120, 180, 240, 300]
    
    for i, target_hue in enumerate(target_hues):
        print(f"Applying color transformation to hue {target_hue}...")
        
        # Additive transformation
        transformed_hsi_add = apply_color_transformation(hsi_img, mask_hsi, target_hue, 'additive')
        transformed_rgb_add = hsi_to_rgb(transformed_hsi_add)
        save_image(os.path.join(out_dir, f"transformed_additive_{target_hue}.jpg"), transformed_rgb_add)
        
        # Multiplicative transformation
        if seed_hsi[0] != 0:
            transformed_hsi_mult = apply_color_transformation(hsi_img, mask_hsi, target_hue, 'multiplicative')
            transformed_rgb_mult = hsi_to_rgb(transformed_hsi_mult)
            save_image(os.path.join(out_dir, f"transformed_multiplicative_{target_hue}.jpg"), transformed_rgb_mult)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original and two example transformations
    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load some transformed images for display
    transformed_add_120 = apply_color_transformation(hsi_img, mask_hsi, 120, 'additive')
    transformed_rgb_120 = cv2.cvtColor(hsi_to_rgb(transformed_add_120), cv2.COLOR_BGR2RGB)
    
    transformed_add_240 = apply_color_transformation(hsi_img, mask_hsi, 240, 'additive')
    transformed_rgb_240 = cv2.cvtColor(hsi_to_rgb(transformed_add_240), cv2.COLOR_BGR2RGB)
    
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(transformed_rgb_120)
    axes[0, 1].set_title('Transformed to Green (120°)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(transformed_rgb_240)
    axes[0, 2].set_title('Transformed to Blue (240°)')
    axes[0, 2].axis('off')
    
    # Show masks used
    axes[1, 0].imshow(mask_rgb * 255, cmap='gray')
    axes[1, 0].set_title('RGB Color Slicing Mask')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_hsi * 255, cmap='gray')
    axes[1, 1].set_title('HSI Color Slicing Mask')
    axes[1, 1].axis('off')
    
    # HSI channels of original
    axes[1, 2].imshow(hsi_img[:, :, 0], cmap='hsv')
    axes[1, 2].set_title('Original Hue Channel')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transformation_results.png"), dpi=150)
    plt.close()
    
    print("\nColor transformation complete!")
    print(f"Results saved in: {out_dir}")
    print("\nTransformation Discussion:")
    print("1. Additive transformation T(c) = c + (ct - cs) shifts all hue values by a constant")
    print("   This preserves relative color relationships and is good for subtle color changes")
    print("2. Multiplicative transformation T(c) = (ct/cs) * c scales hue values")
    print("   This can create more dramatic changes but may not preserve color relationships")
    print("3. For HSI space:")
    print("   - Hue (H): Use additive for natural color shifts, multiplicative for dramatic changes")
    print("   - Saturation (S): Multiplicative is often better to maintain color intensity")
    print("   - Intensity (I): Usually kept unchanged or use additive for brightness adjustments")

if __name__ == "__main__":
    main()