import sys # I use it just to exit from python
import cv2 # import the opencv module
import numpy as np # import numpy module refer to as np
import pytesseract # importy tesseract for OCR module
import torch # this is the part for AI training module pytorch
import json # import json module to manipulate our json files
from torchvision import models # torch for computer vision
import torch.nn as nn # just to access the neural network features of torch easier
from torch.utils.data import Dataset, DataLoader # import torch data manipulation utilities/tools directly to make calling easier

class Datasets(Dataset):
    def __init__(self, image_dir, json_file):
        self.image_dir = image_dir
        with open(json_file, 'r') as f:
            self.labels = json.load(f)
        self.image_names = list(self.labels.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = f"{self.image_dir}/{img_name}"
        
        # Load image and get original dimensions
        image = cv2.imread(img_path)
        orig_h, orig_w = image.shape[:2]
        
        # Resize image to 224x224 (ResNet standard)
        image = cv2.resize(image, (224, 224))
        
        # Normalize colors for PyTorch (0 to 1 scale) and rearrange dimensions
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)) # (Channels, Height, Width)
        
        # Get original 4 corners [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
        points = np.array(self.labels[img_name], dtype=np.float32)
        
        # Squash the coordinates to match the new 224x224 image size
        points[:, 0] = points[:, 0] * (224.0 / orig_w) # scale X
        points[:, 1] = points[:, 1] * (224.0 / orig_h) # scale Y
        
        # Flatten 4 points into 8 numbers
        points = points.flatten()
        
        return torch.tensor(image), torch.tensor(points)

def ourModel():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # resNet18 which is a pretrained AI model for image recognition in torchvision.
    num_features = model.fc.in_features # Look to the Fully Connected (fc) layer, in other words the last layer, and look into the features it is receiving from the previous layer
    model.fc = nn.Linear(num_features, 8) # Create/replace the last layer with a new layer/connectors that inputs in_features and outputs 8 only (what we modify)
    return model # return the adjusted model

def train():
    # Load Up the Data
    data = Datasets('train', 'labels.json') # Setup the data by throwing it into the Dataset class
    loaddata = DataLoader(data, batch_size=4, shuffle=True) # Load the data/(image reference with its correspong json) in batch of four while shuffling it
    
    # Set the settings of the training
    model = ourModel() # Load the our modified ResNet18 model
    lossfunc = nn.MSELoss() # We use Mean Squared Error function to calculate the loss function (how far the prediction from the ground truth)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # We use ADAM (Adaptive Moment Estimation) Optimizer to change the models weights; The model.parameters() gives the weights and loss; lr is the learning rate/how big of a shift can the optimizer make

    #Begin training
    print("Begin Training")
    for epoch in range(50): # Do the training 30 times
        totalLoss = 0 # Initialize the loss for an image from 0
        for images, realCorners in loaddata: 
            optimizer.zero_grad() # resets/cleans the optimizer back to zero
            predCorners = model(images) # Model predicting the 8 coordinates
            loss = lossfunc(predCorners, realCorners) # Calculate the loss
            # Backpropagation | Optimization
            loss.backward() # Calculate the gradients of each weights/neurons based on the loss (the gradient tells the direction of optimization)
            optimizer.step() # Updates the gradients and adjust each weights based on the steps magnitude
            totalLoss += loss.item() # Updates the total loss
        print(f"Epoch {epoch+1}/50 | Error (Loss): {totalLoss/len(loaddata):.2f}")
    torch.save(model.state_dict(), 'corner_finder.pth') # save the trained model
    print("Model is saved!")

def run():
    model = models.resnet18(weights=None) # Load the ResNet18 Model
    model.fc = nn.Linear(model.fc.in_features, 8) # Adjust and change the back of the model nn
    model.load_state_dict(torch.load('corner_finder.pth')) # import/load our trained model
    model.eval() # Choose the weights for inference

    #1. Get image and its specs + resizing
    img = cv2.imread('test.jpg') # load the img we want to manipulate
    h, w = img.shape[:2] # get the image original height and width from .shape which includes all aspects of the image
    resizing = cv2.resize(img, (224, 224)) # resize the photo to fit the model

    #2. Convert so that Pytorch understand
    imgTensor = resizing.astype(np.float32)/255.0 # Change from opencv 255 pixel format to torch tensor decimal for pixel format
    imgTensor = np.transpose(imgTensor, (2,0,1)) # Change the order of placing height, width, channels (0,1,2) in opencv to the pytorch channels, height, width (2, 0, 1)
    imgTensor = torch.tensor(imgTensor).unsqueeze(0) # change it from numpy array to pytorch tensor & add 1 batch dimension at front

    #3. Run the AI model!!!!
    with torch.no_grad(): # No need to track the gradient of each weight/neuron
        predictions = model(imgTensor).numpy()[0] # Get the 8 numbers prediction by the model in numpy arrray form

    #4. Convert the prediction back to the coordinates
    predictedSimple = predictions.reshape(4,2) # convert it to x, y structure in 2d array format for the 4 points
    predictedSimple[:, 0] = predictedSimple[:, 0] * (w/224.0) # scale the x coordinate back to the original based on the width (done by multiplying it with the original width divide by the previously change width)
    predictedSimple[:, 1] = predictedSimple[:, 1] * (h/224.0) # scale the y coordinate back to the original
    
    '''
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
    #5. Warped and Reshape the Image
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

    # Reorder the predicted value
    set = order(predictedSimple) # Order the AI model prediction coordinates: top-left, top-right, bottom-right, bottom-left
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

    # Calculate the matrix that straighten/flatten the detected coordinate to the expected dimension
    mapping = cv2.getPerspectiveTransform(set, rect)
    # Warp the original img to fit the new transformation
    warping = cv2.warpPerspective(img, mapping, (avgWidth, avgHeight))
    cv2.imshow("Warped", warping)
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
                train()
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