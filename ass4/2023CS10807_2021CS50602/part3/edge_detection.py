import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import os
os.makedirs('results', exist_ok=True)

def load_image(image_path):
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img, dtype=np.float64)

def gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def convolve2d(image, kernel):
    return ndimage.convolve(image, kernel, mode='constant', cval=0.0)

def compute_gradients(img_smooth):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)
    
    Gx = convolve2d(img_smooth, sobel_x)
    Gy = convolve2d(img_smooth, sobel_y)
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    gradient_direction = np.arctan2(Gy, Gx)
    
    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_mag, gradient_dir):
    M, N = gradient_mag.shape
    suppressed = np.zeros((M, N), dtype=np.float64)
    angle = gradient_dir * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255
            # Angle 0 (horizontal edge)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_mag[i, j+1]
                r = gradient_mag[i, j-1]
            # Angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_mag[i+1, j-1]
                r = gradient_mag[i-1, j+1]
            # Angle 90 (vertical edge)
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_mag[i+1, j]
                r = gradient_mag[i-1, j]
            # Angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_mag[i-1, j-1]
                r = gradient_mag[i+1, j+1]
            # Keep only local maxima
            if gradient_mag[i, j] >= q and gradient_mag[i, j] >= r:
                suppressed[i, j] = gradient_mag[i, j]
            else:
                suppressed[i, j] = 0
    
    return suppressed

def hysteresis_thresholding(img, low_threshold, high_threshold):
    M, N = img.shape
    strong_edges = img >= high_threshold
    weak_edges = (img >= low_threshold) & (img < high_threshold)
    # Use morphological reconstruction (allowed by assignment)
    # This connects weak edges to strong edges
    result = ndimage.binary_dilation(strong_edges, iterations=0).astype(np.uint8)
    # Manual hysteresis: connect weak edges to strong edges
    # Simple 8-connectivity check
    edge_map = strong_edges.astype(np.uint8)
    
    changed = True
    iterations = 0
    max_iterations = 50
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        new_edge_map = edge_map.copy()
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                if weak_edges[i, j] and not edge_map[i, j]:
                    # Check if any neighbor is a strong edge
                    if np.any(edge_map[i-1:i+2, j-1:j+2]):
                        new_edge_map[i, j] = 1
                        changed = True
        
        edge_map = new_edge_map
    
    return edge_map

def canny_edge_detection(image, sigma=1.5, low_threshold=50, high_threshold=150):
    print(f"Canny parameters: sigma={sigma}, low={low_threshold}, high={high_threshold}")
    # Step 1: Gaussian smoothing
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    gaussian = gaussian_kernel(kernel_size, sigma)
    img_smooth = convolve2d(image, gaussian)
    
    # Step 2: Compute gradients
    gradient_mag, gradient_dir = compute_gradients(img_smooth)
    
    # Step 3: Non-maximum suppression
    nms_result = non_maximum_suppression(gradient_mag, gradient_dir)
    
    # Step 4: Double thresholding
    high_thresh_only = (nms_result >= high_threshold).astype(np.uint8) * 255
    low_thresh_only = (nms_result >= low_threshold).astype(np.uint8) * 255
    
    # Step 5: Edge tracking by hysteresis
    final_edges = hysteresis_thresholding(nms_result, low_threshold, high_threshold)
    
    return {
        'gradient_magnitude': gradient_mag,
        'nms_result': nms_result,
        'high_threshold_only': high_thresh_only,
        'low_threshold_only': low_thresh_only,
        'final_edges': final_edges * 255
    }

def hough_transform(edge_image, rho_resolution=1, theta_resolution=1):
    """Implement Hough transform for line detection"""
    M, N = edge_image.shape
    edge_pixels = np.argwhere(edge_image > 0)
    diagonal = np.sqrt(M**2 + N**2)
    max_rho = int(np.ceil(diagonal))
    rho_bins = int(2 * max_rho / rho_resolution)
    theta_bins = int(180 / theta_resolution)
    accumulator = np.zeros((rho_bins, theta_bins), dtype=np.int32)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))
    
    print(f"Hough transform parameters:")
    print(f"  Rho resolution: {rho_resolution} pixels")
    print(f"  Theta resolution: {theta_resolution} degrees")
    print(f"  Accumulator size: {rho_bins} x {theta_bins}")
    print(f"  Processing {len(edge_pixels)} edge pixels...")

    for y, x in edge_pixels:
        for theta_idx, theta in enumerate(thetas):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = int(np.round((rho + max_rho) / rho_resolution))
            
            if 0 <= rho_idx < rho_bins:
                accumulator[rho_idx, theta_idx] += 1
    
    return accumulator, rho_resolution, theta_resolution, max_rho

def detect_lines(accumulator, edge_image, num_lines=15, rho_res=1, theta_res=1, max_rho=500, nms_window=10):
    """Detect top lines from Hough accumulator with NMS"""
    M, N = edge_image.shape
    accumulator_copy = accumulator.copy()
    
    lines = []
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    
    for _ in range(num_lines):
        max_val = np.max(accumulator_copy)
        if max_val == 0:
            break
            
        max_idx = np.unravel_index(np.argmax(accumulator_copy), accumulator_copy.shape)
        rho_idx, theta_idx = max_idx
        rho = (rho_idx * rho_res) - max_rho
        theta = thetas[theta_idx]
        
        lines.append((rho, theta, max_val))        
        r_start = max(0, rho_idx - nms_window)
        r_end = min(accumulator_copy.shape[0], rho_idx + nms_window + 1)
        t_start = max(0, theta_idx - nms_window)
        t_end = min(accumulator_copy.shape[1], theta_idx + nms_window + 1)
        
        accumulator_copy[r_start:r_end, t_start:t_end] = 0
    
    return lines

def draw_lines(image, lines):
    M, N = image.shape
    output = np.stack([image]*3, axis=-1)
    
    if output.max() <= 1.0:
        output = (output * 255).astype(np.uint8)
    else:
        output = output.astype(np.uint8)
    
    for rho, theta, _ in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        
        if abs(b) > abs(a):
            y0 = int(rho / b)
            y1 = int((rho - a * (N-1)) / b)
            x0, x1 = 0, N-1
        else:
            x0 = int(rho / a)
            x1 = int((rho - b * (M-1)) / a)
            y0, y1 = 0, M-1
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steps = max(dx, dy)
        
        if steps > 0:
            x_inc = (x1 - x0) / steps
            y_inc = (y1 - y0) / steps
            
            x, y = x0, y0
            for _ in range(int(steps) + 1):
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < M and 0 <= xi < N:
                    output[yi, xi] = [0, 255, 0]
                x += x_inc
                y += y_inc
    
    return output

def find_intersections(lines, M, N):
    intersections = []
    
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            rho1, theta1, _ = lines[i]
            rho2, theta2, _ = lines[j]            
            if abs(theta1 - theta2) < 0.05:
                continue
            
            a = np.array([[np.cos(theta1), np.sin(theta1)],
                         [np.cos(theta2), np.sin(theta2)]])
            b = np.array([rho1, rho2])
            
            try:
                point = np.linalg.solve(a, b)
                x, y = point
                if 0 <= x < N and 0 <= y < M:
                    intersections.append((int(x), int(y)))
            except np.linalg.LinAlgError:
                continue
    
    return intersections

def draw_intersections(image, intersections, radius=5):
    output = image.copy()
    
    for x, y in intersections:
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if dx*dx + dy*dy <= radius*radius:
                    yi, xi = y + dy, x + dx
                    if 0 <= yi < output.shape[0] and 0 <= xi < output.shape[1]:
                        output[yi, xi] = [255, 0, 0]
    
    return output

def save_results(original, results, lines, intersections):
    # Save gradient magnitude
    plt.figure(figsize=(10, 8))
    plt.imshow(results['gradient_magnitude'], cmap='gray')
    plt.title('Gradient Magnitude')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('results/1_gradient_magnitude.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save NMS result
    plt.figure(figsize=(10, 8))
    plt.imshow(results['nms_result'], cmap='gray')
    plt.title('Non-Maximum Suppression Result')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('results/2_nms_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save high threshold only
    plt.figure(figsize=(10, 8))
    plt.imshow(results['high_threshold_only'], cmap='gray')
    plt.title('High Threshold Only')
    plt.axis('off')
    plt.savefig('results/3_high_threshold.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save low threshold only
    plt.figure(figsize=(10, 8))
    plt.imshow(results['low_threshold_only'], cmap='gray')
    plt.title('Low Threshold Only')
    plt.axis('off')
    plt.savefig('results/4_low_threshold.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save final edges
    plt.figure(figsize=(10, 8))
    plt.imshow(results['final_edges'], cmap='gray')
    plt.title('Final Canny Edges (After Hysteresis)')
    plt.axis('off')
    plt.savefig('results/5_final_edges.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save lines on original image
    if lines is not None:
        lines_img = draw_lines(original, lines)
        plt.figure(figsize=(10, 8))
        plt.imshow(lines_img)
        plt.title(f'Detected Lines (Top {len(lines)} lines)')
        plt.axis('off')
        plt.savefig('results/7_detected_lines.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save lines with intersections
        if intersections:
            final_img = draw_intersections(lines_img, intersections)
            plt.figure(figsize=(10, 8))
            plt.imshow(final_img)
            plt.title(f'Lines and Intersections ({len(intersections)} points)')
            plt.axis('off')
            plt.savefig('results/8_lines_and_intersections.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print("\nAll results saved in 'results/' directory")

if __name__ == "__main__":
    image_path = 'building1.png'
    original_image = load_image(image_path)
    print(f"Image loaded: {original_image.shape}")
    
    # Part (a): Canny edge detection
    print("\n=== Part (a): Canny Edge Detection ===")
    canny_results = canny_edge_detection(
        original_image,
        sigma=1.5,           # Gaussian width
        low_threshold=50,    # Low threshold
        high_threshold=150   # High threshold
    )
    
    # Part (b): Hough transform
    print("\n=== Part (b): Hough Transform ===")
    accumulator, rho_res, theta_res, max_rho = hough_transform(
        canny_results['final_edges'],
        rho_resolution=1,    # Rho bin width in pixels
        theta_resolution=1   # Theta bin width in degrees
    )
    
    # Save Hough accumulator
    plt.figure(figsize=(12, 8))
    hough_display = np.log(accumulator + 1)  # Log scale for better visibility
    plt.imshow(hough_display, cmap='hot', aspect='auto')
    plt.title('Hough Transform (log scale)')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.colorbar(label='Log(Votes + 1)')
    
    theta_ticks = np.arange(0, accumulator.shape[1], accumulator.shape[1]//6)
    theta_labels = np.linspace(-90, 90, len(theta_ticks)).astype(int)
    plt.xticks(theta_ticks, theta_labels)
    
    plt.savefig('results/6_hough_transform.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Normalize and save as 0-255 image as required
    hough_normalized = ((accumulator - accumulator.min()) / 
                        (accumulator.max() - accumulator.min()) * 255).astype(np.uint8)
    Image.fromarray(hough_normalized).save('results/6_hough_transform_normalized.png')
    
    # Part (c): Detect lines and intersections
    print("\n=== Part (c): Line Detection and Intersections ===")
    lines = detect_lines(
        accumulator,
        canny_results['final_edges'],
        num_lines=15,
        rho_res=rho_res,
        theta_res=theta_res,
        max_rho=max_rho,
        nms_window=10
    )
    
    print(f"\nDetected {len(lines)} lines:")
    for idx, (rho, theta, votes) in enumerate(lines, 1):
        print(f"  Line {idx}: rho={rho:.1f}, theta={np.rad2deg(theta):.1f}°, votes={votes}")
    
    intersections = find_intersections(lines, *original_image.shape)
    print(f"\nFound {len(intersections)} intersection points inside the image")        
    save_results(original_image, canny_results, lines, intersections)
    
    print("\n=== Summary ===")
    print(f"Canny parameters: sigma=1.5, low_threshold=50, high_threshold=150")
    print(f"Hough parameters: rho_res={rho_res}, theta_res={theta_res}")
    print(f"Accumulator size: {accumulator.shape[0]} x {accumulator.shape[1]}")
    print(f"Lines detected: {len(lines)}")
    print(f"Intersections found: {len(intersections)}")