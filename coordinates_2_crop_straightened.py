import cv2
import numpy as np

# Load the image
img = cv2.imread('14a.jpg')

# Define your 4 points (e.g., [[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
# Order: Top-left, top-right, bottom-right, bottom-left
x1= 362
x2= 452
x3= 452
x4= 363
y1= 277
y2= 270
y3= 290
y4= 297
pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

# Calculate width and height of the new cropped image
widthA = np.sqrt(((pts[2][0] - pts[3][0]) ** 2) + ((pts[2][1] - pts[3][1]) ** 2))
widthB = np.sqrt(((pts[1][0] - pts[0][0]) ** 2) + ((pts[1][1] - pts[0][1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((pts[1][0] - pts[2][0]) ** 2) + ((pts[1][1] - pts[2][1]) ** 2))
heightB = np.sqrt(((pts[0][0] - pts[3][0]) ** 2) + ((pts[0][1] - pts[3][1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

# Destination points for a flat rectangle output
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype=np.float32)

# Get the perspective transform matrix and apply it
M = cv2.getPerspectiveTransform(pts, dst)
cropped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

cv2.imwrite('cropped_output.jpg', cropped)
