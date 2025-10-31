#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if the correct number of arguments is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <directory1> <directory2> ..."
    echo "Example : ./mergeCleanData.sh keras_dataset_weld/class_clean/ keras_dataset_positive/class_clean/ keras_dataset_negative_balanced/class_clean/"
    exit 1
fi

# Directory where symbolic links will be created
outputdir="keras_dataset_merged/class_clean"

# Create the output directory if it doesn't exist
mkdir -p "$outputdir"

# Initialize counter for numbering scheme
count=1

# Loop through each directory provided as input
for directory in "$@"; do
    # Check if the directory exists
    if [ ! -d "$directory" ]; then
        echo "Error: Directory '$directory' not found."
        continue
    fi
    
    # Loop through each file in the directory
    for file in "$directory"/*; do
        # Check if it's a file
        if [ -f "$file" ]; then
            # Extract file extension
            ext="${file##*.}"
            # Create symbolic link with new numbering scheme
            ln -s "../../$file" "$outputdir/clean_image_$count.$ext"
            # Increment counter
            ((count++))
        fi
    done
done

echo "Symbolic links created in '$outputdir' with new numbering scheme."

