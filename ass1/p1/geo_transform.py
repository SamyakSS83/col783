import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees
from typing import Tuple, List, Any

def ensure_dir(path: str) -> str:
    if not path or path.strip() == "":
        return os.getcwd()
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path

def load_image_color(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def four_point_transform(image: np.ndarray, pts: np.ndarray, interpolation) -> np.ndarray:
    rect = order_points_clockwise(np.asarray(pts, dtype="float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),
                                 flags=interpolation,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    return warped

def rotate_image(img: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype="float32")
    if pts.shape[0] != 4:
        raise ValueError("rotate_image requires exactly 4 corner points")

    # Order the points consistently: tl, tr, br, bl
    tl, tr, br, bl = pts

    mid_top = (tl + tr) / 2
    mid_bottom = (bl + br) / 2
    dx = mid_top[0] - mid_bottom[0]
    dy = mid_bottom[1] - mid_top[1]

    # Calculate the rotation angle for the transformation.
    angle_rad = atan2(dy, dx)
    angle_deg = degrees(angle_rad)
    rotation_angle = (90 - angle_deg)

    # Get the center of the image for rotation
    (h, w) = img.shape[:2]
    (cx, cy) = (w // 2, h // 2)

    # Get the 2D rotation matrix from OpenCV
    M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)

    # Apply the rotation to the entire image
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))

    # Calculate the new positions of our corner points
    rotated_pts = cv2.transform(np.array([pts]), M)[0]
    return rotated, rotated_pts

def save_image(path: str, img: np.ndarray) -> None:
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

def order_points_clockwise(pts: Any) -> np.ndarray:
    # Orders the 4 corners in the order: top-left, top-right, bottom-right, bottom-left
    pts = np.asarray(pts, dtype="float32")
    if pts.shape[0] != 4:
        raise ValueError("order_points_clockwise requires exactly 4 points")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def pick_document_corners(img_path: str, n_points: int) -> np.ndarray:
    # Show the image and let the user click n_points (using matplotlib ginput)
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise RuntimeError(f"cv2.imread failed for {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 10))
    plt.imshow(rgb)
    plt.title(f"Click {n_points} points to choose the document corners. Press Enter when done.")
    plt.axis('off')
    pts = plt.ginput(n=n_points, timeout=0)
    plt.close()
    raw_pts = [tuple(map(float, p)) for p in pts]
    if len(raw_pts) != 4:
        raise RuntimeError("You must click exactly 4 points (top-left, top-right, bottom-right, bottom-left).")
    # order points clockwise
    ordered = order_points_clockwise(raw_pts)
    return ordered

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python geo_transform.py <input_image> [output_dir]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) == 3 else os.path.join(os.path.dirname(__file__), "results")
    ensure_dir(out_dir)

    img = load_image_color(in_path)
    img_vis = img.copy()

    # Picking the 4 corners manually
    corners = pick_document_corners(in_path, 4)

    # Rotate the image to align the document vertically
    rotated, rotated_pts = rotate_image(img, corners)
    save_image(os.path.join(out_dir, "rotated.jpg"), rotated)

    rotated_pts_ordered = order_points_clockwise(rotated_pts)
    # Nearest Neighbour Interpolation
    warped_nn = four_point_transform(rotated, rotated_pts_ordered, interpolation=cv2.INTER_NEAREST)
    save_image(os.path.join(out_dir, "scanned_nearest.jpg"), warped_nn)

    # Bilinear Interpolation
    warped_bl = four_point_transform(rotated, rotated_pts_ordered, interpolation=cv2.INTER_LINEAR)
    save_image(os.path.join(out_dir, "scanned_bilinear.jpg"), warped_bl)

    # Show zoomed-in regions (say, top-left 200x200 area for comparison)
    zoom_nn = warped_nn[:200, :200]
    zoom_bl = warped_bl[:200, :200]
    save_image(os.path.join(out_dir, "zoom_nearest.jpg"), zoom_nn)
    save_image(os.path.join(out_dir, "zoom_bilinear.jpg"), zoom_bl)

if __name__ == "__main__":
    main()
