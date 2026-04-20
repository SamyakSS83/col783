# HDR Histogram Equalization (No Quantization)

This implements histogram equalization for HDR images with arbitrary real-valued intensities. The goal is to map intensities to a uniform distribution over a target range [a, b] (here [0, 256)).

## Method (for the report)
Given an image with N pixels and real intensities:

1) Build the empirical CDF F of intensities
- Flatten all pixel values and sort them: x_(1) ≤ x_(2) ≤ … ≤ x_(N)
- Assign ranks r(i) = i, so F(x_(i)) = r(i) / N

2) Target mapping to uniform range [a, b]
- Define T(x) = a + (b − a) · F(x)
- This spreads values uniformly across [a, b]

3) Interpolate to original pixels
- Multiple pixels may share the same intensity; keep unique x values and their mapped T(x)
- For each original pixel value, linearly interpolate between the nearest unique x values
- Apply per-channel for color images

Notes:
- No quantization or binning is used; we operate directly on floating point values
- The mapping is monotone and preserves ordering

Mathematical properties (concise):
- If g is strictly increasing, then ranks are preserved: x_i < x_j ⇒ g(x_i) < g(x_j)
- Thus F_g(g(x)) = F(x) and histogram equalization after g equals equalizing the original
- Applying equalization twice does nothing further (idempotent)

## Python (try.py)
Requirements: Python 3, OpenCV (cv2), NumPy

Run:
- python histogram_equalization.py nave.hdr

Output:
- results/hdr_original.jpg
- results/hdr_equalized.jpg

## C++ (hdr_equalize.cpp)
Requirements: OpenCV (core, imgproc, imgcodecs, photo), C++17, OpenMP optional

Build (example):
- g++ -O3 -fopenmp hdr_equalize.cpp -o hdr_equalize $(pkg-config --cflags --libs opencv4)

Run:
- ./hdr_equalize nave.hdr

Output:
- results/hdr_original.jpg
- results/hdr_equalized.jpg
