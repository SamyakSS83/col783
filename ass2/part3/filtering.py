import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Create results folder
RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created results directory: {RESULTS_DIR}")

def save_image(image, filename, cmap='gray'):
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {filepath}")

def save_figure(fig, filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"Saved: {filepath}")

def load_image(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float64)
    return img_array

def compute_fft(image):
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    return fft_shifted

def visualize_spectrum(fft_shifted, title="Fourier Spectrum", save_name=None):
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log(1 + magnitude)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(log_magnitude, cmap='gray')
    plt.colorbar(label='Log Magnitude')
    plt.title(title)
    plt.xlabel('Frequency u')
    plt.ylabel('Frequency v')
    
    if save_name:
        save_figure(plt.gcf(), save_name)
    
    return log_magnitude

def manual_spike_selection(log_magnitude, num_spikes=4):
    plt.figure(figsize=(12, 10))
    plt.imshow(log_magnitude, cmap='gray')
    plt.title('Click on halftone frequency spikes')
    plt.colorbar(label='Log Magnitude')
    
    points = plt.ginput(n=num_spikes, timeout=0, show_clicks=True)
    plt.close()
    spike_locs = [(int(y), int(x)) for x, y in points]
    print(f"\nSelected spike locations (row, col):")

    return spike_locs

def analyze_dot_spacing(spike_locs, image_shape):
    center = (image_shape[0] // 2, image_shape[1] // 2)
    
    print("\n--- Spatial Domain Analysis ---")
    print(f"Image shape: {image_shape}")
    print(f"Spectrum center: {center}")
    
    # Calculate distances from center
    for i, (r, c) in enumerate(spike_locs):
        dr = r - center[0]
        dc = c - center[1]
        distance = np.sqrt(dr**2 + dc**2)
        angle = np.arctan2(dr, dc) * 180 / np.pi
        
        # Spatial period = image_size / frequency
        if distance > 0:
            period_pixels = image_shape[0] / distance
        else:
            period_pixels = float('inf')
        
        print(f"\nSpike {i+1}:")
        print(f"  Offset from center: ({dr}, {dc})")
        print(f"  Distance: {distance:.2f} pixels")
        print(f"  Angle: {angle:.2f} degrees")
        print(f"  Corresponding spatial period: ~{period_pixels:.2f} pixels")
    
    # Find displacement vectors between spikes (for pattern analysis)
    if len(spike_locs) >= 2:
        print("\n--- Pattern Displacement Vectors ---")
        for i in range(len(spike_locs)-1):
            for j in range(i+1, len(spike_locs)):
                dr = spike_locs[j][0] - spike_locs[i][0]
                dc = spike_locs[j][1] - spike_locs[i][1]
                print(f"  Spike {i+1} to Spike {j+1}: ({dr}, {dc})")
    
    return center

def generate_symmetric_spikes(spike_locs, center, image_shape):
    all_spikes = set()
    for spike in spike_locs:
        all_spikes.add(spike)
    
    for r, c in spike_locs:
        # Reflect through center: (center - (spike - center)) = (2*center - spike)
        reflected_r = 2 * center[0] - r
        reflected_c = 2 * center[1] - c
        
        if 0 <= reflected_r < image_shape[0] and 0 <= reflected_c < image_shape[1]:
            all_spikes.add((reflected_r, reflected_c))
    
    all_spikes_list = sorted(list(all_spikes))
    
    print("\n--- Generated Symmetric Spike Pattern ---")
    print(f"Original spikes: {len(spike_locs)}")
    print(f"Total spikes (including symmetric): {len(all_spikes_list)}")
    print("All spike locations:")
    for i, spike in enumerate(all_spikes_list):
        offset = (spike[0] - center[0], spike[1] - center[1])
        print(f"  Spike {i+1}: {spike}, offset: {offset}")

    for i, loc in enumerate(all_spikes_list):
        print(f"  Spike {i+1}: {loc}")
    
    return all_spikes_list

# Part (b): Notch filter with Gaussian or Butterworth profile
def create_notch_filter(shape, all_spike_locs, width, filter_type='gaussian', n=2):
    rows, cols = shape
    filter_mask = np.ones((rows, cols), dtype=np.float64)
    r, c = np.ogrid[:rows, :cols]
    
    for spike_r, spike_c in all_spike_locs:
        dist = np.sqrt((r - spike_r)**2 + (c - spike_c)**2)
        
        if filter_type == 'gaussian':
            # Gaussian notch: 1 - exp(-(dist^2)/(2*width^2))
            notch = 1 - np.exp(-(dist**2) / (2 * width**2))
        elif filter_type == 'butterworth':
            # Butterworth notch: 1 / (1 + (width/dist)^(2n))
            dist = np.maximum(dist, 1e-10)
            notch = 1.0 / (1.0 + (width / dist)**(2*n))
        
        filter_mask *= notch
    
    return filter_mask

def apply_filter(fft_shifted, filter_mask):
    filtered_fft = fft_shifted * filter_mask
    return filtered_fft

def reconstruct_image(filtered_fft):
    fft_ishifted = np.fft.ifftshift(filtered_fft)
    img_filtered = np.fft.ifft2(fft_ishifted)
    img_filtered = np.real(img_filtered)
    return img_filtered

# Part (c): Ideal notch filter
def create_ideal_notch_filter(shape, all_spike_locs, width):
    rows, cols = shape
    filter_mask = np.ones((rows, cols), dtype=np.float64)
    
    r, c = np.ogrid[:rows, :cols]

    for spike_r, spike_c in all_spike_locs:
        dist = np.sqrt((r - spike_r)**2 + (c - spike_c)**2)
        filter_mask[dist <= width] = 0
    
    return filter_mask

# Part (d): Optimum notch filtering
def optimum_notch_filter(image, fft_shifted, all_spike_locs, neighborhood_size, initial_filter):
    rows, cols = image.shape
    
    # Step 1: Apply initial notch filter to estimate noise η
    filtered_fft_initial = fft_shifted * initial_filter
    g = image
    noise_estimate = image - reconstruct_image(filtered_fft_initial)
    
    print("\n--- Optimum Notch Filtering ---")
    print(f"Using {len(all_spike_locs)} spike locations for filtering")
    print(f"Neighborhood size: {neighborhood_size}x{neighborhood_size}")
    print(f"Noise estimate η statistics:")
    print(f"  Mean: {np.mean(noise_estimate):.6f}")
    print(f"  Std: {np.std(noise_estimate):.6f}")
    print(f"  Variance: {np.var(noise_estimate):.6f}")
    
    # Step 2: Compute local statistics for optimum weight
    half_size = neighborhood_size // 2
    
    # Pad images for neighborhood processing
    g_padded = np.pad(g, half_size, mode='reflect')
    noise_padded = np.pad(noise_estimate, half_size, mode='reflect')
    
    # Initialize output
    f_optimum = np.zeros_like(g, dtype=np.float64)
    weight_map = np.zeros_like(g, dtype=np.float64)
    
    # For display: store first computed values
    first_cov = None
    first_var = None
    first_w = None
    
    # Process each pixel
    print("Computing optimum weights for each pixel...")
    for i in range(rows):
        if i % 50 == 0:
            print(f"  Processing row {i}/{rows}")
        for j in range(cols):
            i_pad = i + half_size
            j_pad = j + half_size
            
            g_neighborhood = g_padded[i_pad-half_size:i_pad+half_size+1, 
                                      j_pad-half_size:j_pad+half_size+1]
            n_neighborhood = noise_padded[i_pad-half_size:i_pad+half_size+1,
                                          j_pad-half_size:j_pad+half_size+1]
            
            cov_g_n = np.mean((g_neighborhood - np.mean(g_neighborhood)) * 
                             (n_neighborhood - np.mean(n_neighborhood)))
            var_n = np.var(n_neighborhood)
            
            # Compute optimum weight w = cov(g,η) / var(η)
            if var_n > 1e-10:
                w = cov_g_n / var_n
            else:
                w = 0

            w = np.clip(w, 0, 1)

            if first_cov is None:
                first_cov = cov_g_n
                first_var = var_n
                first_w = w
            
            # Compute filtered value: f = g - w*η
            f_optimum[i, j] = g[i, j] - w * noise_estimate[i, j]
            weight_map[i, j] = w
    
    print(f"\nSample computation (first pixel):")
    print(f"  cov(g, η) = {first_cov:.6f}")
    print(f"  var(η) = {first_var:.6f}")
    print(f"  w = {first_w:.6f}")
    
    print(f"\nWeight statistics across image:")
    print(f"  Mean weight: {np.mean(weight_map):.6f}")
    print(f"  Std weight: {np.std(weight_map):.6f}")
    print(f"  Min weight: {np.min(weight_map):.6f}")
    print(f"  Max weight: {np.max(weight_map):.6f}")
    
    intermediate = {
        'cov': first_cov,
        'var': first_var,
        'weight': first_w,
        'weight_map': weight_map
    }
    
    return f_optimum, noise_estimate, intermediate

# Visualization functions
def plot_results(original, filtered, title_prefix, spike_locs=None, log_spectrum=None, save_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Halftone Image')
    axes[0].axis('off')
    
    axes[1].imshow(filtered, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'{title_prefix} - Filtered Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_name:
        save_figure(fig, save_name)
    
    plt.show()

def plot_spectrum_with_filter(log_spectrum, filter_mask, all_spike_locs, title, save_name=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original spectrum with spike markers
    axes[0].imshow(log_spectrum, cmap='gray')
    axes[0].set_title(f'Original Spectrum\n({len(all_spike_locs)} spike locations marked)')
    if all_spike_locs:
        for r, c in all_spike_locs:
            circle = Circle((c, r), radius=10, fill=False, color='red', linewidth=2)
            axes[0].add_patch(circle)
    axes[0].set_xlabel('u')
    axes[0].set_ylabel('v')
    
    # Filter mask
    axes[1].imshow(filter_mask, cmap='gray')
    axes[1].set_title(f'{title} Filter Mask')
    axes[1].set_xlabel('u')
    axes[1].set_ylabel('v')
    
    # Filtered spectrum
    filtered_spectrum = log_spectrum * filter_mask
    axes[2].imshow(filtered_spectrum, cmap='gray')
    axes[2].set_title('Filtered Spectrum')
    axes[2].set_xlabel('u')
    axes[2].set_ylabel('v')
    
    plt.tight_layout()
    
    if save_name:
        save_figure(fig, save_name)
    
    plt.show()

def main(image_path):
    img = load_image(image_path)
    print(f"Image shape: {img.shape}")
    save_image(img, "00_original_image.png")
    
    # Part (a): Fourier spectrum and spike analysis
    print("\n=== Part (a): Fourier Spectrum Analysis ===")
    fft_shifted = compute_fft(img)
    log_magnitude = visualize_spectrum(fft_shifted, "Part (a): Fourier Spectrum (Centered)", 
                                      save_name="01_fft_spectrum.png")
    plt.show()
    spike_locs = manual_spike_selection(log_magnitude, num_spikes=12)
    center = analyze_dot_spacing(spike_locs, img.shape)
    all_spike_locs = generate_symmetric_spikes(spike_locs, center, img.shape)
    fig_spikes = plt.figure(figsize=(10, 8))
    plt.imshow(log_magnitude, cmap='gray')
    plt.colorbar(label='Log Magnitude')
    plt.title(f'Fourier Spectrum with All {len(all_spike_locs)} Spike Locations')
    for r, c in all_spike_locs:
        circle = Circle((c, r), radius=10, fill=False, color='red', linewidth=2)
        plt.gca().add_patch(circle)
    plt.xlabel('u')
    plt.ylabel('v')
    save_figure(fig_spikes, "02_spectrum_all_spikes_marked.png")
    plt.close()
    
    # Part (b): Gaussian/Butterworth notch filter
    print("\n=== Part (b): Notch Filter (Gaussian) ===")
    width_b = 25
    filter_gaussian = create_notch_filter(img.shape, all_spike_locs, width_b, 
                                         filter_type='gaussian')
    
    plot_spectrum_with_filter(log_magnitude, filter_gaussian, all_spike_locs, 
                             f"Gaussian (width={width_b})",
                             save_name="03_gaussian_notch_spectrum.png")
    save_image(filter_gaussian, "04_gaussian_filter_mask.png")
    
    filtered_fft_b = apply_filter(fft_shifted, filter_gaussian)
    img_filtered_b = reconstruct_image(filtered_fft_b)
    
    save_image(img_filtered_b, "05_gaussian_filtered_image.png")
    
    plot_results(img, img_filtered_b, f"Part (b): Gaussian Notch (width={width_b})",
                save_name="06_gaussian_comparison.png")
    
    # Part (c): Ideal notch filter
    print("\n=== Part (c): Ideal Notch Filter ===")
    width_c = width_b
    filter_ideal = create_ideal_notch_filter(img.shape, all_spike_locs, width_c)
    
    plot_spectrum_with_filter(log_magnitude, filter_ideal, all_spike_locs,
                             f"Ideal (radius={width_c})",
                             save_name="07_ideal_notch_spectrum.png")
    
    save_image(filter_ideal, "08_ideal_filter_mask.png")
    
    filtered_fft_c = apply_filter(fft_shifted, filter_ideal)
    img_filtered_c = reconstruct_image(filtered_fft_c)
    
    # Save filtered image
    save_image(img_filtered_c, "09_ideal_filtered_image.png")
    
    plot_results(img, img_filtered_c, f"Part (c): Ideal Notch (radius={width_c})",
                save_name="10_ideal_comparison.png")
    
    # Part (d): Optimum notch filtering
    print("\n=== Part (d): Optimum Notch Filtering ===")
    neighborhood_size = 31  # Should span at least 2 periods of dot pattern
    
    img_optimum, noise_est, intermediate = optimum_notch_filter(
        img, fft_shifted, all_spike_locs, neighborhood_size, filter_gaussian
    )
    
    # Save intermediate results
    save_image(noise_est, "11_noise_estimate.png")
    save_image(intermediate['weight_map'], "12_weight_map.png", cmap='viridis')
    save_image(img_optimum, "13_optimum_filtered_image.png")
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noise_est, cmap='gray')
    axes[0, 1].set_title(f'Estimated Noise η\n(from notch filter)')
    axes[0, 1].axis('off')
    
    im = axes[1, 0].imshow(intermediate['weight_map'], cmap='viridis')
    axes[1, 0].set_title(f'Weight Map w\n(mean={np.mean(intermediate["weight_map"]):.3f})')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    axes[1, 1].imshow(img_optimum, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title(f'Part (d): Optimum Filtered Result\n(neighborhood={neighborhood_size}x{neighborhood_size})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_figure(fig, "14_optimum_all_results.png")
    plt.show()
    
    # Compare all methods
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Halftone')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_filtered_b, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Gaussian Notch Filter')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(img_filtered_c, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Ideal Notch Filter')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_optimum, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Optimum Notch Filter')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_figure(fig, "15_all_methods_comparison.png")
    plt.show()
    
    print("\n=== Processing Complete ===")
    print(f"All results saved in '{RESULTS_DIR}/' directory")

if __name__ == "__main__":
    image_path = "halftone.png"
    main(image_path)