import cv2 # import the opencv module
import numpy as np # import numpy module refer to as np

#1. Detect the edge
img = cv2.imread('test1.jpg') # load the img we want to manipulate
img = cv2.resize(img, (800, 600)) # resize the photo to fit the screen
grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
blurred = cv2.GaussianBlur(grayed, (5, 5), 0) # Blurring the img to remove unwanted artifacts and noise
edges = cv2.Canny(blurred, 50, 150) # Detects edges

#2. Find the contours (outlines)
# RETR_LIST: get all contours (without family relationship)
# CHAIN_APPROX_SIMPLE: compresses straight lines, recognising the corner points
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Sort the top 5 largest contours by area (Since the object would probably be big)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

board_contour = None # contour result will be put here

# Loop to find the one with exactly 4 corners from the top 5
for contour in contours:
    perimeter = cv2.arcLength(contour, True) #Calculate the perimeter
    
    # Simplify jagged contours into a smooth line to reduce unnecessary corners or contours (0.02 is the multiplier/tolerance)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    # 4 points will probably be the board, so choose this
    if len(approx) == 4:
        board_contour = approx
        break

'''
# Visualizatio of contours
if board_contour is not None:
    cv2.drawContours(img, [board_contour], -1, (0, 255, 0), 3) # Apply green color on the contour
    
    # Draw red circles on the 4 corners
    for point in board_contour:
        x, y = point[0]
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

# Show the final img
cv2.imshow('Detected Whiteboard', img)
'''

# A helper function to ensure our 4 corners are always sorted correctly
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has the smallest x+y sum; Bottom-right has the largest x+y sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has the smallest y-x difference; Bottom-left has the largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

# [Previous code: load img, find board_contour]

if board_contour is not None:
    # The contour has extra brackets (shape is 4,1,2). Strip them to get 4 pairs of x,y coordinates
    pts = board_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate the maximum width of the new flattened img using the Pythagorean theorem
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate the maximum height
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the exact destination points of our perfect rectangle
    dst = np.array([
        [0, 0],                             # Top-Left
        [maxWidth - 1, 0],                  # Top-Right
        [maxWidth - 1, maxHeight - 1],      # Bottom-Right
        [0, maxHeight - 1]                  # Bottom-Left
    ], dtype="float32")

    # Compute the transformation matrix that maps the crooked corners to the flat ones
    matrix = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply the mathematical warp to the original high-resolution img
    warped = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))

    # Display the snapped-straight result
    cv2.imshow("Original", img)
    cv2.imshow("Flattened Board", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.waitKey(0) # Wait for a key input
cv2.destroyAllWindows() # Close all windows