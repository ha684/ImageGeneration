import os
from PIL import Image

def remove_incorrect_srgb_profile(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                with Image.open(input_path) as img:
                    if "icc_profile" in img.info:
                        img.info["icc_profile"] = None  # Remove the incorrect ICC profile
                    img.save(output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# Example usage
input_folder = r"D:\Workspace\python_code\TextRecognitionDataGenerator\trdg\images_out"  # Replace with your folder containing the images
output_folder = r"D:\Workspace\python_code\TextRecognitionDataGenerator\trdg\images_out1"  # Replace with your desired output folder
remove_incorrect_srgb_profile(input_folder, output_folder)
def remove(folder):
    paths = os.listdir(output_folder)
    for path in paths:
        if path.endswith('.txt'):
            os.remove(os.path.join(output_folder,path))
import json

def json_to_txt(json_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as txt_file:
        for image_name, label in json_data.items():
            txt_file.write(f"{image_name}\t{label}\n")

import os

def check_and_fix_txt_file(txt_file_path):
    corrected_lines = []
    
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        # Split the line by the first tab character
        parts = line.split('\t', 1)
        
        if len(parts) != 2:
            continue
        else:
            corrected_lines.append(line)

    # Write the corrected lines back to the file (optional)
    corrected_file_path = os.path.splitext(txt_file_path)[0] + "_corrected.txt"
    with open(corrected_file_path, 'w', encoding='utf-8') as file:
        file.writelines(corrected_lines)
    
    print(f"Finished checking and fixing the file. Corrected file saved as {corrected_file_path}.")

# # Example usage
# txt_file_path = r'D:\Workspace\python_code\TextRecognitionDataGenerator\trdg\images_out1\label_train1.txt'
# check_and_fix_txt_file(txt_file_path)

