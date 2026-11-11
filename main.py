import sys # I use it just to exit from python
import cv2 # import the opencv module
import numpy as np # import numpy module refer to as np
import pytesseract # importy tesseract for OCR module
import torch # this is the part for AI training module pytorch
import torch.nn as nn

def train():
    ...

def run():
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

    #3. Warped and Reshape the Image
    # Sort 4 corner to top-left, top-right, bottom-right, bottom-left
    def order(corner):
        set = np.zeros((4, 2), dtype="float32") # default rect
        
        # Top-left has the smallest x+y sum; Bottom-right has the largest x+y sum
        s = corner.sum(axis=1) # sum the axis
        set[0] = corner[np.argmin(s)] # min sum at top-left
        set[2] = corner[np.argmax(s)] # max sum at bottom-right
        # Top-right has the smallest y-x difference; Bottom-left has the largest
        diff = np.diff(corner, axis=1) # find the difference
        set[1] = corner[np.argmin(diff)] # min difference at top-right
        set[3] = corner[np.argmax(diff)] # mac difference at bottom-left
        
        return set

    if board_contour is not None:
        corner = board_contour.reshape(4, 2) # Change contour previously 4, 1, 2 (4 points, 1 basically nothing, 2 axis[x and y]) structure to 4, 2 (4 points and their 2 axis)
        set = order(corner) # Order it to have an order of: top-left, top-right, bottom-right, bottom-left
        (tl, tr, br, bl) = set # Assign the order to their variables

        # Maximum width of the new flattened img using the Pythagorean theorem
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)) # Using the difference in x and y axis of the bottom horizontal points, we can achieve width by summing both squareroot of the difference.
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)) # Do the same but for the top part
        avgWidth = int((widthA+widthB)/2) # from both width, find the maximum

        # Maximum height
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)) # Do the same pythagorean method but for height at the right
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)) # This one is for the left
        avgHeight = int((heightA+heightB)/2) # Find the highest one out of both

        # Find the new = coordinates of the estimated rectangle
        rect = np.array([
            [0, 0], # Top-Left
            [avgWidth - 1, 0], # Top-Right
            [avgWidth - 1, avgHeight - 1], # Bottom-Right
            [0, avgHeight - 1] # Bottom-Left
        ], dtype="float32")

        # Map the old corners to the new flat ones
        mapping = cv2.getPerspectiveTransform(set, rect)
        # Warp the original img to fit the new transformation
        warping = cv2.warpPerspective(img, mapping, (avgWidth, avgHeight))

    graywarp = cv2.cvtColor(warping, cv2.COLOR_BGR2GRAY) # make the  warped img grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(graywarp)
    clean_board = cv2.adaptiveThreshold(
        clahe_image, 
        255,                              # Maximum value (pure white background)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # Use a Gaussian curve to weigh neighboring pixels
        cv2.THRESH_BINARY,                # Output black text on a white background
        21,                               # Block Size: The size of the local neighborhood (must be odd)
        10                                # Constant (C): Subtracted from the mean. Acts as a tuning knob.
    )

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    custom_config = r'--oem 3 --psm 11'
    extracted_text = pytesseract.image_to_string(clahe_image, config=custom_config)
    print("\n=== EXTRACTED WHITEBOARD NOTES ===")
    print(extracted_text)

    cv2.imshow("OG", img)
    cv2.imshow("Flattened (Uneven Lighting)", graywarp)
    cv2.imshow("CLAHE (Contrast Fixed)", clahe_image)
    cv2.imshow("Final Clean Board", clean_board)
    cv2.waitKey(0) # Wait for a key input
    cv2.destroyAllWindows() # Close all windows

# The def main is just for the menu option
def main():
    print("WELCOME TO STRAIGHTIFY!")
    while(1):
        print("Choose What to do!")
        print("(1) Train the Edge Detection Model\n(2) Run the program\n(3) Exit")
        num = int(input("I Choose: ")) 
        match num:
            case 1: # retrain for the AI (Please Don't)
                print("Let's Train.")
            case 2: # run our pre-trained program
                print("Running...")
                run()
            case 3: # exit
                print("Bye!")
                break
            case _: # you give weird stuff!
                print("What?")
                continue
    sys.exit()

if __name__ == "__main__": # Just to initialize things properly and call main()
    main()