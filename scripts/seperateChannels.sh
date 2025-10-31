#!/bin/bash

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "ImageMagick is not installed. Please install it first."
    exit 1
fi

# Check if an image file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_image>"
    exit 1
fi

# Check if the input file exists
if [ ! -f "$1" ]; then
    echo "Input file not found: $1"
    exit 1
fi

# Extract channels
input_file="$1"
filename="${input_file%.*}"

# Extract Red channel
convert "$input_file" -channel R -separate "${filename}_red.png"

# Extract Green channel
convert "$input_file" -channel G -separate "${filename}_green.png"

# Extract Blue channel
convert "$input_file" -channel B -separate "${filename}_blue.png"

# Extract Alpha channel
#convert "$input_file" -channel A -separate "${filename}_alpha.png"

convert "$input_file" -channel RGBA -separate gray:image_%d.raw


# Convert raw files to PNG
for file in image_*.raw; do
    convert -depth 8 -size "$(identify -format '%wx%h' "$input_file")" rgba:"$file" "${filename}_${file}.png"
done

# Clean up raw files
rm image_*.raw

echo "Channels extracted successfully."

