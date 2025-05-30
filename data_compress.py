# reducing the 50gb dataset into 30gb one. 

import os

# Define file paths
input_file = r"C:\Users\iampr\OneDrive\Desktop\major work\managing_the_dataset\combined_tamil_dataset.txt"
output_file = "C:/Users/iampr/OneDrive/Desktop/major work/managing_the_dataset/compressed_data.txt"

# Define the target size (30GB)
target_size = 30 * 1024 * 1024 * 1024  # Convert GB to bytes

# Define a condition to remove lines (e.g., remove every 5th line)
def should_remove(line_num):
    return line_num % 5 == 0  # Modify this condition as needed

# Process the file line by line
with open(input_file, "r", encoding="utf-8", errors="ignore") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    line_count = 0
    bytes_written = 0

    for line in infile:
        line_count += 1

        # Remove lines based on condition
        if should_remove(line_count):
            continue

        # Write line to the new file
        outfile.write(line)
        bytes_written += len(line.encode("utf-8"))

        # Stop when we reach the target size
        if bytes_written >= target_size:
            break

print(f"Processing complete. New file size: {os.path.getsize(output_file) / (1024 * 1024 * 1024):.2f} GB")

