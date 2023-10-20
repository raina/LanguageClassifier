# Rename multiple files in a directory or folder based on original name
# For language-detection dataset this has to be run twice per folder due to outlier with two underscores

# importing os module
import os
import re

# Function to rename multiple files
def main():

    folder = "data/train"
    for count, filename in enumerate(os.listdir(folder)):
        match = re.search(r'^[^_]*_', filename) # match up to first underscore
        if match:
            shortName = re.sub(r'^[^_]*_', '', filename); # removes matched section
        else:
            shortName = filename
        
        dst = f"{shortName}{str(count)}.jpg" # Adding numeric ID like count prevents overwritten files
        src =f"{folder}/{filename}"
        dst =f"{folder}/{dst}"
        
        os.rename(src, dst)

# Driver Code
if __name__ == '__main__':
    
    # Calling main() function
    main()
