#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..
 

# Directory where symbolic links will be created
merged_dataset="binary_dataset"
clean_outputdir="$merged_dataset/class_clean"
outputdir="$merged_dataset/class_defect"

# Create the output directory if it doesn't exist
mkdir -p "$outputdir"

#Link Clean data
cd $merged_dataset
ln -s  "../keras_dataset/class_clean"
cd ..

# Initialize counter for numbering scheme
count=1

# Loop through each directory provided as input
for directory in "keras_dataset/class_PositiveDentClassA" "keras_dataset/class_PositiveDentClassB" "keras_dataset/class_PositiveDentClassC" "keras_dataset/class_NegativeDentClassA" "keras_dataset/class_NegativeDentClassB" "keras_dataset/class_NegativeDentClassC" "keras_dataset/class_WeldingClassA"; do
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
            ln -s "../../$file" "$outputdir/defect_image_$count.$ext"
            # Increment counter
            ((count++))
        fi
    done
done

echo "Symbolic links created in '$outputdir' with new numbering scheme."

