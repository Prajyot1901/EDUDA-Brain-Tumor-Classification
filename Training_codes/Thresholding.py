import os
import cv2
import numpy as np

def binary_threshold(image):
    """
    Applies binary thresholding to a grayscale image.
    Non-zero values → 255, zero remains 0.
    """
    binary_image = np.where(image > 35, 255, 0).astype(np.uint8)
    return binary_image

def process_folder(input_folder, output_folder):
    """
    Processes all grayscale images in the input folder,
    applies binary thresholding, and saves them to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image in grayscale mode
            gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"Skipping {filename} (not a valid image).")
                continue

            # Apply binary thresholding
            binary = binary_threshold(gray)

            # Save the binary image
            cv2.imwrite(output_path, binary)
            print(f"Processed: {filename}")

slices = ['Coronal', 'Sagittal','Axial']
for i in slices:
    input_folder = r"C:\Users\Sapna\Downloads\Transfer learning\Transfer learning\UDA_MRI\Slice_Separation_Training_Testing\Training\\" + i
    output_folder = r"C:\Users\Sapna\Downloads\Transfer learning\Transfer learning\UDA_MRI\Slice_Separation_Training_Testing\Training_binary\\" + i
    process_folder(input_folder, output_folder)
for i in slices:
   input_folder = r"C:\Users\Sapna\Downloads\Transfer learning\Transfer learning\UDA_MRI\Slice_Separation_Training_Testing\Testing_Source\\" + i
   output_folder = r"C:\Users\Sapna\Downloads\Transfer learning\Transfer learning\UDA_MRI\Slice_Separation_Training_Testing\Testing_Source_binary\\" + i
   process_folder(input_folder, output_folder)
for i in slices:
    input_folder = r"C:\Users\Sapna\Downloads\Transfer learning\Transfer learning\UDA_MRI\Slice_Separation_Training_Testing\Testing_Target\\" + i
    output_folder = r"C:\Users\Sapna\Downloads\Transfer learning\Transfer learning\UDA_MRI\Slice_Separation_Training_Testing\Testing_Target_binary\\" + i
    process_folder(input_folder, output_folder)