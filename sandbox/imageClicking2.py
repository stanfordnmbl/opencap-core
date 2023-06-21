# function to click on points on an image and save them

# import the necessary packages
import cv2
import os

# %%
refPts = []
def click_and_save(event, x, y, flags, param):  
    # grab references to the global variables
    global refPts
    if event == cv2.EVENT_LBUTTONDOWN:
        refPts.append((x,y))

# %% 
def pickPointsFromImage(imagePath,nPts):
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(imagePath)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_save)
    print('Click on ' + str(nPts) + ' points or press q when finished.')
    
    # keep looping until the 'q' key is pressed
    ptLength = 0
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        
        if len(refPts) > ptLength:
            image = cv2.circle(image, refPts[-1], radius=2, color=(0, 255, 0), thickness=-1)
        
        if key == ord("q") or len(refPts) >nPts:
            refPts.pop(-1) # last click is just to exit
            cv2.destroyAllWindows()
            break

    
basePath = 'C:/Users/suhlr/Downloads'
imagePath = os.path.join(basePath,'testImage.jpg')

pickPointsFromImage(