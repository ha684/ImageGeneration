# Define the input and output file paths
input_file_path = r'D:\Workspace\python_code\TextRecognitionDataGenerator\trdg\images_out\label_test.txt'
output_file_path = r'D:\Workspace\python_code\TextRecognitionDataGenerator\trdg\images_out\label_test1.txt'

# Open the input file for reading and output file for writing
with open(input_file_path, 'r',encoding='utf-8') as infile, open(output_file_path, 'w',encoding='utf-8') as outfile:
    for line in infile:
        line = line.strip()
        parts = line.split('\t')

        if len(parts) > 2:

            corrected_line = parts[0] + '\t' + ' '.join(parts[1:])
        elif len(parts) == 2:
            corrected_line = line
        else:
            continue
        
        outfile.write(corrected_line + '\n')

print("Processing complete. Check the output file for the corrected data.")
