import cv2
import numpy as np

# Function to convert gray background to white
def gray_to_white(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set the threshold just below white (255). Assuming gray is in the range of 220-240.
    threshold = 220

    # Create a mask where gray pixels are turned white
    mask = gray > threshold

    # Convert the mask to a three channel image
    mask_3_channel = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Convert all masked pixels to white
    img[mask_3_channel == 1] = 255

    # Save the edited image
    edited_image_path = 'edited_image.png'
    cv2.imwrite(edited_image_path, img)
    
    return edited_image_path

# Path to the image we're assuming exists
image_path = 'hmm.png'
edited_image_path = gray_to_white(image_path)
edited_image_path
