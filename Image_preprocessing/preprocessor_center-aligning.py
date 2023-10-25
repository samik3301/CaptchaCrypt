import cv2
import os
def find_word_and_margins(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale

    #Binary thresholding to distinctly process intensity values better
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Finding the starting and ending points of the word
    start, end = 0, 0
    left_margin = 0
    right_margin = 0
    height, width = thresh.shape
    #print(height, width)
    #print(thresh[:,0])

    for i in range(width-1):
        if 0 in thresh[:,i]: #black pixel intensity detected
            start = i #the starting index - marks the starting letter of the word
            break

    for i in range(width-1,0,-1):
        if 0 in thresh[:,i]: #black pixel intensity detected
            end = i #the ending index -  marks the ending letter of the word
            break

    left_margin = start 
    right_margin = image.shape[1]-end

    word_region = image[:,start:end] # Crop the word region
    # Calculate the required margin to make both sides equal
    required_margin = max(left_margin, right_margin)
    # Resize the image with the calculated margin and align it in the middle
    resized_image = cv2.copyMakeBorder(word_region, 0, 0, required_margin, required_margin, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return resized_image

def resizing_image(img_path):
    img = cv2.imread(img_path)
    width = 200 #captcha original width - according to the keras implementation : width=200
    left_margin = (img.shape[1] - width) // 2
    right_margin = img.shape[1] - left_margin
    resized_image = img[:, left_margin:right_margin]
    return resized_image


if __name__ == "__main__":
    input_image_path = '/Users/samik/Desktop/Programming/CaptchaCrypt/Image_preprocessing/test/п3Зе6u.png'  # Change this to the path of your input image
    output_image_path = '/Users/samik/Desktop/Programming/CaptchaCrypt/Image_preprocessing/margin.png'  # Change this to the desired output image path
    result_image = find_word_and_margins(input_image_path)
    #print(result_image.shape)
    resized_image = resizing_image(output_image_path)
    cv2.imwrite(output_image_path, resized_image)
    #print(resized_image.shape)
    
    #Save all the processed images from the ultimate captcha data directory to a new directory under data [data/processed_data], with same filename
    #Tested this script with a sample captcha, it should work fine for all now- gn