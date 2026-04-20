import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_means_manual(image, k, manual_points=None):
    if manual_points is not None and len(manual_points) == k:
        means = np.array([image[p[0], p[1], :] for p in manual_points], dtype=float)
    else:
        h, w, _ = image.shape
        random_indices = np.random.choice(h * w, k, replace=False)
        random_rows = random_indices // w
        random_cols = random_indices % w
        means = image[random_rows, random_cols, :].astype(float)
    
    return means

def assign_pixels_to_clusters(pixels, means):
    n_pixels = pixels.shape[0]
    k = means.shape[0]
    labels = np.zeros(n_pixels, dtype=int)
    
    for i in range(n_pixels):
        min_dist = float('inf')
        best_cluster = 0
        
        for j in range(k):
            dist = euclidean_distance(pixels[i], means[j])
            if dist < min_dist:
                min_dist = dist
                best_cluster = j
        
        labels[i] = best_cluster
    
    return labels

def update_means(pixels, labels, k):
    new_means = np.zeros((k, 3))
    
    for j in range(k):
        cluster_pixels = pixels[labels == j]
        if len(cluster_pixels) > 0:
            new_means[j] = np.mean(cluster_pixels, axis=0)
        else:
            new_means[j] = pixels[np.random.randint(len(pixels))]
    
    return new_means

def kmeans_segmentation(image, k, max_iterations=100, tolerance=1e-4, manual_points=None):
    h, w, c = image.shape
    
    pixels = image.reshape(-1, 3).astype(float)    
    means = initialize_means_manual(image, k, manual_points)
    
    print(f"Starting k-means with k={k}")
    print(f"Initial means:\n{means}")
    
    for iteration in range(max_iterations):
        labels = assign_pixels_to_clusters(pixels, means)
        new_means = update_means(pixels, labels, k)
        mean_shift = np.max(np.abs(new_means - means))
        print(f"Iteration {iteration + 1}: max mean shift = {mean_shift:.6f}")
        
        if mean_shift < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
        means = new_means
    
    segmented_pixels = means[labels]
    segmented_image = segmented_pixels.reshape(h, w, c).astype(np.uint8)
    labels = labels.reshape(h, w)
    
    return labels, means, segmented_image

def find_boundaries(labels):
    h, w = labels.shape
    boundaries = np.zeros((h, w), dtype=bool)
    
    for i in range(h):
        for j in range(w - 1):
            if labels[i, j] != labels[i, j + 1]:
                boundaries[i, j] = True
                boundaries[i, j + 1] = True
    
    for i in range(h - 1):
        for j in range(w):
            if labels[i, j] != labels[i + 1, j]:
                boundaries[i, j] = True
                boundaries[i + 1, j] = True
    
    return boundaries

def visualize_segmentation(original_image, segmented_image, boundaries, output_path):
    result = segmented_image.copy()
    result[boundaries] = [255, 255, 255]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(segmented_image)
    axes[1].set_title('Segmented Image (Cluster Means)')
    axes[1].axis('off')
    
    axes[2].imshow(result)
    axes[2].set_title('Segmentation with Boundaries')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")

def rgb_to_lab(rgb_image):
    rgb = rgb_image.astype(float) / 255.0    
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    r = rgb_linear[:, :, 0]
    g = rgb_linear[:, :, 1]
    b = rgb_linear[:, :, 2]
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    X = X / 0.95047
    Y = Y / 1.00000
    Z = Z / 1.08883
    def f(t):
        delta = 6/29
        return np.where(t > delta**3, t**(1/3), t/(3*delta**2) + 4/29)
    
    fx = f(X)
    fy = f(Y)
    fz = f(Z)
    
    L = 116 * fy - 16
    A = 500 * (fx - fy)
    B = 200 * (fy - fz)
    
    return np.stack([L, A, B], axis=2)

def compute_gradient_magnitude(image):
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    h, w = gray.shape
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            window = gray[i-1:i+2, j-1:j+2]
            gx[i, j] = np.sum(window * sobel_x)
            gy[i, j] = np.sum(window * sobel_y)
    
    return np.sqrt(gx**2 + gy**2)

def initialize_cluster_centers_slic(lab_image, s):
    h, w = lab_image.shape[:2]
    centers = []
    
    for y in range(s // 2, h, s):
        for x in range(s // 2, w, s):
            l, a, b = lab_image[y, x]
            centers.append([float(l), float(a), float(b), float(x), float(y)])
    
    return np.array(centers)

def perturb_centers_to_lowest_gradient(centers, lab_image, gradient):
    h, w = gradient.shape
    perturbed = centers.copy()
    
    for i in range(len(centers)):
        cx, cy = int(centers[i, 3]), int(centers[i, 4])
        min_grad = gradient[cy, cx]
        best_x, best_y = cx, cy
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if gradient[ny, nx] < min_grad:
                        min_grad = gradient[ny, nx]
                        best_x, best_y = nx, ny
        
        perturbed[i, 3] = best_x
        perturbed[i, 4] = best_y
        perturbed[i, :3] = lab_image[best_y, best_x]
    
    return perturbed

def slic_distance(pixel_lab, pixel_xy, center_lab, center_xy, s, m=10):
    d_lab = np.sqrt(np.sum((pixel_lab - center_lab) ** 2))
    d_xy = np.sqrt(np.sum((pixel_xy - center_xy) ** 2))
    return np.sqrt((d_lab / m) ** 2 + (d_xy / s) ** 2)

def assign_pixels_slic(lab_image, centers, s, m=10):
    h, w = lab_image.shape[:2]
    labels = -np.ones((h, w), dtype=int)
    distances = np.full((h, w), np.inf)
    
    for k, center in enumerate(centers):
        cx, cy = int(center[3]), int(center[4])
        center_lab = center[:3]
        center_xy = center[3:]
        
        y_min = max(0, cy - s)
        y_max = min(h, cy + s)
        x_min = max(0, cx - s)
        x_max = min(w, cx + s)
        
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel_lab = lab_image[y, x]
                pixel_xy = np.array([x, y])
                dist = slic_distance(pixel_lab, pixel_xy, center_lab, center_xy, s, m)
                
                if dist < distances[y, x]:
                    distances[y, x] = dist
                    labels[y, x] = k
    
    return labels

def update_centers_slic(lab_image, labels, num_centers):
    h, w = lab_image.shape[:2]
    new_centers = np.zeros((num_centers, 5))
    counts = np.zeros(num_centers)
    
    for y in range(h):
        for x in range(w):
            label = labels[y, x]
            if label >= 0:
                new_centers[label, :3] += lab_image[y, x]
                new_centers[label, 3] += x
                new_centers[label, 4] += y
                counts[label] += 1
    
    for k in range(num_centers):
        if counts[k] > 0:
            new_centers[k] /= counts[k]
    
    return new_centers

def enforce_connectivity_slic(labels, min_size=10):
    h, w = labels.shape
    new_labels = labels.copy()
    visited = np.zeros((h, w), dtype=bool)
    
    def bfs(start_y, start_x, label):
        component = []
        queue = [(start_y, start_x)]
        visited[start_y, start_x] = True
        
        while queue:
            y, x = queue.pop(0)
            component.append((y, x))
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and 
                    not visited[ny, nx] and labels[ny, nx] == label):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        
        return component
    
    for y in range(h):
        for x in range(w):
            if not visited[y, x]:
                label = labels[y, x]
                component = bfs(y, x, label)
                
                if len(component) < min_size:
                    neighbor_labels = []
                    for cy, cx in component:
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if (0 <= ny < h and 0 <= nx < w and 
                                labels[ny, nx] != label):
                                neighbor_labels.append(labels[ny, nx])
                    
                    if neighbor_labels:
                        new_label = max(set(neighbor_labels), key=neighbor_labels.count)
                        for cy, cx in component:
                            new_labels[cy, cx] = new_label
    
    return new_labels

def slic_superpixels(image, s, m=10, max_iterations=10):
    print(f"\nComputing SLIC superpixels with s={s}, m={m}")
    
    lab_image = rgb_to_lab(image)
    gradient = compute_gradient_magnitude(image)
    centers = initialize_cluster_centers_slic(lab_image, s)
    centers = perturb_centers_to_lowest_gradient(centers, lab_image, gradient)
    
    print(f"Initialized {len(centers)} superpixel centers")
    
    for iteration in range(max_iterations):
        labels = assign_pixels_slic(lab_image, centers, s, m)
        new_centers = update_centers_slic(lab_image, labels, len(centers))
        
        shift = np.max(np.abs(new_centers - centers))
        print(f"Iteration {iteration + 1}: max center shift = {shift:.6f}")
        
        if shift < 0.5:
            print(f"Converged after {iteration + 1} iterations")
            break
        centers = new_centers
    
    labels = enforce_connectivity_slic(labels)
    return labels, centers

def build_graph_for_segmentation(image, fg_pixels, bg_pixels, lambda_param=100):
    from scipy.sparse import lil_matrix
    
    h, w = image.shape[:2]
    n_pixels = h * w
    
    fg_colors = np.array([image[y, x] for y, x in fg_pixels])
    bg_colors = np.array([image[y, x] for y, x in bg_pixels])
    
    fg_mean = np.mean(fg_colors, axis=0)
    bg_mean = np.mean(bg_colors, axis=0)
    capacity = lil_matrix((n_pixels + 2, n_pixels + 2), dtype=np.int32)
    
    def pixel_to_node(y, x):
        return 2 + y * w + x
    
    SCALE = 1000
    for y in range(h):
        for x in range(w):
            node = pixel_to_node(y, x)
            color = image[y, x].astype(float)
            
            fg_dist = np.sum((color - fg_mean) ** 2)
            bg_dist = np.sum((color - bg_mean) ** 2)
            
            if (y, x) in fg_pixels:
                capacity[0, node] = SCALE * 100
                capacity[node, 1] = 0
            elif (y, x) in bg_pixels:
                capacity[0, node] = 0
                capacity[node, 1] = SCALE * 100
            else:
                bg_likelihood = int(SCALE / (1 + bg_dist / 10000))
                fg_likelihood = int(SCALE / (1 + fg_dist / 10000))
                capacity[0, node] = max(1, bg_likelihood)
                capacity[node, 1] = max(1, fg_likelihood)
    
    for y in range(h):
        for x in range(w):
            node = pixel_to_node(y, x)
            color = image[y, x].astype(float)
            
            if x < w - 1:
                neighbor = pixel_to_node(y, x + 1)
                neighbor_color = image[y, x + 1].astype(float)
                color_diff = np.sum((color - neighbor_color) ** 2)
                weight = int(lambda_param * np.exp(-color_diff / 2000))
                weight = max(1, weight)
                capacity[node, neighbor] = weight
                capacity[neighbor, node] = weight
            
            if y < h - 1:
                neighbor = pixel_to_node(y + 1, x)
                neighbor_color = image[y + 1, x].astype(float)
                color_diff = np.sum((color - neighbor_color) ** 2)
                weight = int(lambda_param * np.exp(-color_diff / 2000))
                weight = max(1, weight)
                capacity[node, neighbor] = weight
                capacity[neighbor, node] = weight
    
    return capacity.tocsr()

def graph_cut_segmentation(image, fg_rect, bg_rect, lambda_param=100):
    try:
        from scipy.sparse.csgraph import maximum_flow
    except ImportError:
        print("Error: scipy not available")
        return np.zeros(image.shape[:2], dtype=int)
    
    h, w = image.shape[:2]
    
    fg_y1, fg_x1, fg_y2, fg_x2 = fg_rect
    bg_y1, bg_x1, bg_y2, bg_x2 = bg_rect
    
    fg_pixels = set([(y, x) for y in range(fg_y1, fg_y2) for x in range(fg_x1, fg_x2)])
    bg_pixels = set([(y, x) for y in range(bg_y1, bg_y2) for x in range(bg_x1, bg_x2)])
    
    print(f"Building graph: {len(fg_pixels)} fg, {len(bg_pixels)} bg samples")
    
    capacity = build_graph_for_segmentation(image, fg_pixels, bg_pixels, lambda_param)
    
    print("Computing maximum flow...")
    flow_result = maximum_flow(capacity, 0, 1)
    print(f"Max flow value: {flow_result.flow_value}")
    
    residual = capacity - flow_result.flow
    segmentation = np.zeros((h, w), dtype=int)
    
    def pixel_to_node(y, x):
        return 2 + y * w + x
    
    def node_to_pixel(node):
        if node < 2:
            return None
        idx = node - 2
        return (idx // w, idx % w)
    
    visited = np.zeros(h * w + 2, dtype=bool)
    queue = [0]
    visited[0] = True
    
    while queue:
        u = queue.pop(0)
        neighbors = residual[u].nonzero()[1]
        
        for v in neighbors:
            if not visited[v] and residual[u, v] > 0:
                visited[v] = True
                queue.append(v)
                
                pixel = node_to_pixel(v)
                if pixel:
                    y, x = pixel
                    if 0 <= y < h and 0 <= x < w:
                        segmentation[y, x] = 1
    
    print(f"Foreground pixels: {np.sum(segmentation):,}")
    return segmentation

def morphological_gradient(image):
    h, w = image.shape
    dilated = np.zeros_like(image)
    eroded = np.ones_like(image) * 255
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            dilated[i, j] = np.max(image[i-1:i+2, j-1:j+2])
            eroded[i, j] = np.min(image[i-1:i+2, j-1:j+2])
    
    return dilated - eroded

def h_minima_transform(image, h):
    return np.minimum(image + h, 255)

def find_regional_minima(image):
    h, w = image.shape
    minima = np.ones((h, w), dtype=bool)
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = image[i, j]
            window = image[i-1:i+2, j-1:j+2]
            if np.any(window < center):
                minima[i, j] = False    
    labels = np.zeros((h, w), dtype=int)
    label = 1
    visited = np.zeros((h, w), dtype=bool)
    
    for i in range(h):
        for j in range(w):
            if minima[i, j] and not visited[i, j]:
                queue = [(i, j)]
                visited[i, j] = True
                labels[i, j] = label
                
                while queue:
                    y, x = queue.pop(0)
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < h and 0 <= nx < w and 
                                minima[ny, nx] and not visited[ny, nx]):
                                visited[ny, nx] = True
                                labels[ny, nx] = label
                                queue.append((ny, nx))
                label += 1
    
    return minima, labels

def watershed_segmentation(gradient_image, markers=None, h_min=10):
    print("Computing watershed segmentation...")
    if h_min > 0:
        gradient_image = h_minima_transform(gradient_image, h_min)
    
    if markers is None:
        _, markers = find_regional_minima(gradient_image)
    
    h, w = gradient_image.shape
    labels = markers.copy()
    pixels = []
    for i in range(h):
        for j in range(w):
            if markers[i, j] > 0:
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        if (0 <= ni < h and 0 <= nj < w and 
                            labels[ni, nj] == 0):
                            pixels.append((gradient_image[ni, nj], ni, nj))
    
    pixels.sort()    
    processed = set()
    
    while pixels:
        _, y, x = pixels.pop(0)
        
        if (y, x) in processed or labels[y, x] > 0:
            continue
        
        neighbor_labels = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                    neighbor_labels.append(labels[ny, nx])
        
        if neighbor_labels:
            unique_labels = set(neighbor_labels)
            if len(unique_labels) == 1:
                labels[y, x] = neighbor_labels[0]
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w and 
                            labels[ny, nx] == 0 and (ny, nx) not in processed):
                            pixels.append((gradient_image[ny, nx], ny, nx))
                pixels.sort()
            else:
                labels[y, x] = -1
        
        processed.add((y, x))
    
    print(f"Found {np.max(labels)} watershed basins")
    return labels

def compute_rgb_gradient_magnitude(image):
    gradients = []
    for c in range(3):
        grad = compute_gradient_magnitude(image[:, :, c])
        gradients.append(grad)
    return np.sqrt(np.mean([g**2 for g in gradients], axis=0))

def parta(image_path='input_image.jpg', k=5):
    print("\n" + "="*70)
    print("PART 5a: K-means Color Segmentation")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Please provide a valid image path from Berkeley Segmentation Dataset")
        return
    
    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"Loaded image: {image.shape}")
    
    labels, means, segmented = kmeans_segmentation(image, k, max_iterations=50)
    boundaries = find_boundaries(labels)
    
    visualize_segmentation(
        image, segmented, boundaries,
        f'results/part5a_segmentation_k{k}.png'
    )
    
    Image.fromarray(segmented).save(f'results/part5a_segmented_k{k}.png')
    
    boundary_overlay = image.copy()
    boundary_overlay[boundaries] = [255, 255, 255]
    Image.fromarray(boundary_overlay).save(f'results/part5a_boundaries_k{k}.png')
    
    print("\n" + "="*70)
    print("RESULTS FOR PART 5a")
    print("="*70)
    print(f"\nNumber of clusters (k): {k}")
    print(f"\nFinal cluster means (RGB values):")
    for i, mean in enumerate(means):
        print(f"  Cluster {i}: R={mean[0]:.2f}, G={mean[1]:.2f}, B={mean[2]:.2f}")
    
    print(f"\nPixels per cluster:")
    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = labels.size
    for cluster_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        print(f"  Cluster {cluster_id}: {count:,} pixels ({percentage:.2f}%)")
    
    print("\nOutput files saved:")
    print(f"  - part5a_segmentation_k{k}.png (three-panel visualization)")
    print(f"  - part5a_segmented_k{k}.png (segmented image with cluster means)")
    print(f"  - part5a_boundaries_k{k}.png (boundaries highlighted in white)")

def partb(image_path='input_image.jpg', s=15, m=10):
    print("\n" + "="*70)
    print("PART 5b: SLIC Superpixels (Optional)")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"Loaded image with shape: {image.shape}")
    
    labels, centers = slic_superpixels(image, s, m)
    
    segmented = np.zeros_like(image)
    for label_id in range(len(centers)):
        mask = labels == label_id
        if np.any(mask):
            segmented[mask] = np.mean(image[mask], axis=0)
    
    boundaries = find_boundaries(labels)
    
    visualize_segmentation(
        image, segmented, boundaries,
        f'results/part5b_slic_s{s}_m{m}.png'
    )
    
    Image.fromarray(segmented).save(f'results/part5b_superpixels_s{s}_m{m}.png')
    
    print("\n" + "="*70)
    print("RESULTS FOR PART 5b")
    print("="*70)
    print(f"\nSuperpixel spacing (s): {s} pixels")
    print(f"Compactness parameter (m): {m}")
    print(f"Number of superpixels: {len(centers)}")
    print(f"Average superpixel size: {image.shape[0] * image.shape[1] / len(centers):.1f} pixels")
    
    print("\nOutput files saved:")
    print(f"  - part5b_slic_s{s}_m{m}.png (three-panel visualization)")
    print(f"  - part5b_superpixels_s{s}_m{m}.png (superpixel segmentation)")

def partc(image_path='input_image.jpg'):
    print("\n" + "="*70)
    print("PART 5c: Graph Cut Segmentation (Optional)")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    image = np.array(Image.open(image_path).convert('RGB'))
    h, w = image.shape[:2]
    print(f"Loaded image with shape: {image.shape}")
    
    fg_rect = (h//4, w//4, 3*h//4, 3*w//4)
    bg_rect = (0, 0, h//8, w//8)
    
    print(f"\nForeground rect: {fg_rect}")
    print(f"Background rect: {bg_rect}")
    
    max_size = 300
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_small = np.array(Image.fromarray(image).resize((new_w, new_h)))
        
        fg_rect = tuple(int(x * scale) for x in fg_rect)
        bg_rect = tuple(int(x * scale) for x in bg_rect)
        
        segmentation = graph_cut_segmentation(image_small, fg_rect, bg_rect)
        segmentation = np.array(Image.fromarray((segmentation * 255).astype(np.uint8)).resize((w, h)))
        segmentation = (segmentation > 127).astype(int)
    else:
        segmentation = graph_cut_segmentation(image, fg_rect, bg_rect)
    
    # Visualize
    result = np.zeros_like(image)
    result[segmentation == 1] = [0, 255, 0]
    result[segmentation == 0] = [255, 0, 0]
    
    boundaries = find_boundaries(segmentation)
    result[boundaries] = [255, 255, 255]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(segmentation, cmap='gray')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    axes[2].imshow(result)
    axes[2].set_title('FG (Green) / BG (Red)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/part5c_graphcut.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("RESULTS FOR PART 5c")
    print("="*70)
    print(f"Foreground pixels: {np.sum(segmentation == 1):,} ({100*np.sum(segmentation == 1)/segmentation.size:.2f}%)")
    print(f"Background pixels: {np.sum(segmentation == 0):,} ({100*np.sum(segmentation == 0)/segmentation.size:.2f}%)")
    
    print("\nOutput files saved:")
    print(f"  - part5c_graphcut.png")

def partd(image_path='input_image.jpg', h_min=10):
    print("\n" + "="*70)
    print("PART 5d: Watershed Segmentation (Optional)")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"Loaded image with shape: {image.shape}")
    
    print("\nSmoothing image...")
    from scipy.ndimage import gaussian_filter
    image_smooth = np.zeros_like(image)
    for c in range(3):
        image_smooth[:, :, c] = gaussian_filter(image[:, :, c].astype(float), sigma=2)
    image_smooth = image_smooth.astype(np.uint8)
    
    print("Computing gradient magnitude...")
    gradient = compute_rgb_gradient_magnitude(image_smooth)
    gradient = (gradient / gradient.max() * 255).astype(np.uint8)
    
    # Watershed
    labels = watershed_segmentation(gradient, h_min=h_min)
    
    # Visualize
    num_basins = np.max(labels)
    segmented = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label = labels[i, j]
            if label == -1:
                segmented[i, j] = [255, 255, 255]
            elif label > 0:
                mask = labels == label
                segmented[i, j] = np.mean(image[mask], axis=0)
    
    watershed_lines = (labels == -1)
    result = segmented.copy()
    result[watershed_lines] = [255, 0, 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gradient, cmap='gray')
    axes[0, 1].set_title('Gradient')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(segmented)
    axes[1, 0].set_title('Watershed Basins')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(result)
    axes[1, 1].set_title('With Watersheds (Red)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'results/part5d_watershed_h{h_min}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("RESULTS FOR PART 5d")
    print("="*70)
    print(f"h-minima parameter: {h_min}")
    print(f"Number of watershed basins: {num_basins}")
    print(f"Number of watershed pixels: {np.sum(watershed_lines):,}")
    
    print("\nOutput files saved:")
    print(f"  - part5d_watershed_h{h_min}.png")

def main():
    print("="*70)
    print("ASSIGNMENT 4 - QUESTION 5: IMAGE SEGMENTATION")
    print("="*70)
    
    image_path = '42049.png'
    
    if not os.path.exists(image_path):
        print(f"\nWARNING: Image not found at '{image_path}'")
        return
    
    try:
        parta(image_path, k=5)
    except Exception as e:
        print(f"\nError in Part 5a: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        partb(image_path, s=15, m=10)
    except Exception as e:
        print(f"\nError in Part 5b: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        partc(image_path)
    except Exception as e:
        print(f"\nError in Part 5c: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        partd(image_path, h_min=10)
    except Exception as e:
        print(f"\nError in Part 5d: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print("\nCheck the 'results/' directory for all results.")


if __name__ == "__main__":
    main()