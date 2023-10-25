from PIL import Image, ImageChops
import os

# Set the path to the folder containing your images
input_folder = "/Users/samik/Desktop/Programming/CaptchaCrypt/Image_preprocessing/test"
output_folder = "/Users/samik/Desktop/Programming/CaptchaCrypt/Image_preprocessing/output_save"
desired_width = 200  # Change this to your desired width

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        # Open the image using Pillow
        with Image.open(os.path.join(input_folder, filename)) as img:
            # Convert the image to grayscale for better edge detection
            img_gray = img.convert("L")

            # Calculate a threshold to distinguish white space from content
            threshold = 200

            # Crop the image to remove white space on the left
            left = 0
            for x in range(img_gray.width):
                if img_gray.getpixel((x, img_gray.height // 2)) < threshold:
                    left = x
                else:
                    break

            # Crop the image to remove white space on the right
            right = img_gray.width
            for x in range(img_gray.width - 1, -1, -1):
                if img_gray.getpixel((x, img_gray.height // 2)) < threshold:
                    right = x
                else:
                    break
                
            # Crop the image using the calculated left and right values
            img = img.crop((left, 0, right, img.height))

            # Calculate the new height to maintain aspect ratio
            new_height = int((desired_width / img.width) * img.height)

            # Resize the image to the desired width and height
            img = img.resize((desired_width, new_height), Image.ANTIALIAS)

            # Extract the filename without the extension
            base_filename = os.path.splitext(filename)[0]

            # Save the modified image with the old image's filename
            output_path = os.path.join(output_folder, f"{base_filename}.jpg")  # You can change the extension as needed
            img.save(output_path)

print("Smart image cropping and resizing complete.")
