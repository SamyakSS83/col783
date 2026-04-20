# Geometric Transformation

Geometrically transforms the image of a document to make it vertically aligned and then perform perspective transformation using two methods - Nearest Neighbour Interpolation and Bilinear Interpolation, to create a transformed image that contains as little background as possible.

Usage:
- python geo_transform.py <input_image> [output_image]
  - Defaults output to results/ folder

Output:
- results/rotated.jpg: this contains only the rotated image without the perspective transformation
- results/scanned_bilinear.jpg: this contains the final image transformed using bilinear transformation
- results/scanned_nearest.jpg: this contains the final image transformed using nearest neighbour interpolation transformation
- results/zoom_bilinear: this contains the zoomed parts of the scanned_bilinear image
- results/zoom_nearest: this contains the zoomed parts of the scanned_nearest image

Dependencies:
- Python 3, OpenCV (cv2), NumPy


## Method

1) Detecting the corners of the document
- We first detect the corners of the document by selecting the coordinates manually using the ginput() function from the matplotlib library. 
- Here we prompt the user with the image and ask them to select 4 corners of the document.

2) Rotating the image
- After detecting the coordinates of the corners of the document, we order them in a clockwise manner - top-left, top-right, bottom-left, and bottom-right
- We then use the vector going from top-middle coordinate to the bottom-middle coordinate to perform the rotation and aim to make it vertical. 
- We figure out the original angle and hence the rotation angle
- Using this rotation angle we fetch the rotation matrix from OpenCV and hence apply the same on the original image to get the rotated transformed image.
- We have saved the image for reference in the results folder by the name "rotated.jpg"

3) 