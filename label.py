import cv2
import os
import json
import glob

# --- CONFIGURATION ---
# Create a folder called 'training_images' next to this script and put your DocUNet photos inside
IMAGE_FOLDER = 'train'
OUTPUT_JSON = 'labels.json'

points = []
labels = {}

def click_event(event, x, y, flags, param):
    global points, img_copy
    
    # Listen for Left Mouse Clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            # Draw a red dot where you clicked
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Labeling", img_copy)

# Find all JPG/PNG images in the folder
image_paths = glob.glob(os.path.join(IMAGE_FOLDER, '*.*'))

for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        continue
        
    # Resize down so it fits on your monitor while labeling
    # (We save the coordinates relative to this resized image)
    img = cv2.resize(img, (800, 600))
    img_copy = img.copy()
    points = []
    
    cv2.imshow("Labeling", img_copy)
    cv2.setMouseCallback("Labeling", click_event)
    
    print(f"Click the 4 corners of {os.path.basename(image_path)} (Top-Left, Top-Right, Bottom-Right, Bottom-Left).")
    
    # Wait until 4 points are clicked or user presses 'q' to skip
    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4:
            labels[os.path.basename(image_path)] = points
            print("Saved!")
            break
        elif key == ord('q'):  # Press 'q' to skip an image
            print("Skipped.")
            break

cv2.destroyAllWindows()

# Save all the clicked coordinates to a JSON file for PyTorch
with open(OUTPUT_JSON, 'w') as f:
    json.dump(labels, f, indent=4)

print(f"\nLabeling complete! Coordinates saved to {OUTPUT_JSON}.")