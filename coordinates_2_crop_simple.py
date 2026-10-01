import cv2
import numpy as np

# Load the image
img = cv2.imread('14.jpg')

# Define your 4 points (e.g., [[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
# Order: Top-left, top-right, bottom-right, bottom-left
x1= 334
x2= 402
x3= 403
x4= 335
y1= 206
y2= 198
y3= 212
y4= 221
pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

# Get the minimum and maximum X and Y coordinates to find the straight bounding box
x_min, y_min = np.min(pts, axis=0)
x_max, y_max = np.max(pts, axis=0)

# Convert coordinates to integers (required for pixel indexing)
x_min, y_min = int(x_min), int(y_min)
x_max, y_max = int(x_max), int(y_max)

# Ensure coordinates stay within the original image boundaries
x_min, y_min = max(0, x_min), max(0, y_min)
y_max, x_max = min(img.shape[0], y_max), min(img.shape[1], x_max)

# Crop the upright rectangular region directly using NumPy slicing [ymin:ymax, xmin:xmax]
cropped = img[y_min:y_max, x_min:x_max]

# Save the resulting cropped image
cv2.imwrite('cropped_straight.jpg', cropped)