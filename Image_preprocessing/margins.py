import cv2
import numpy as np

def find_word_and_margins(image_path,min_whitespace_threshold=20):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Find the starting and ending points of the word
    start, end = 0,0
    left_margin = 0
    right_margin = 0
    for i in range(image.shape[1]):
        if i==0:
            start = i

    left_margin = start 

    for i in range(image.shape[1],0,-1):
        if i==0:
            end = i

    right_margin = image.shape[1]-end

    #now finding out which one is larger, left_margin or right_margin
    diff=0
    if right_margin>left_margin:
        diff = right_margin- left_margin
    else:
        diff = left_margin- right_margin

    word_length = end-start

    
        

    '''
    in_whitespace = False  # Track if we are in a whitespace region
    for i in range(thresh.shape[1]):
        if np.any(thresh[:, i] == 0):
            if in_whitespace:
                if i - start > min_whitespace_threshold:
                    end = i
                    break
            else:
                in_whitespace = True
                start = i
        else:
            in_whitespace = False
    '''
    

    # Crop the word region
    word_region = image[:, start:end]

    '''
    # Find the minimum margin on both sides
    left_margin = start
    right_margin = image.shape[1] - end
    '''
    # Calculate the required margin to make both sides equal
    required_margin = max(left_margin, right_margin)
    
    # Resize the image with the calculated margin and align it in the middle
    resized_image = cv2.copyMakeBorder(word_region, 0, 0, required_margin, required_margin, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return resized_image

if __name__ == "__main__":
    input_image_path = '/Users/samik/Desktop/Programming/CaptchaCrypt/Image_preprocessing/test/п3Зе6u.png'  # Change this to the path of your input image
    output_image_path = 'n33e6u.png'  # Change this to the desired output image path

    result_image = find_word_and_margins(input_image_path,min_whitespace_threshold=20)

    # Save the result to an output image file
    cv2.imwrite(output_image_path, result_image)
