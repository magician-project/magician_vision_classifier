import cv2
import numpy as np
import sys

def debayerPolarImage(image):
    """
    De-bayer a polarization image into 4 separate monochrome channels.

    Extracts the 0°, 45°, 90°, and 135° polarization channels from a Bayer-like
    2x2 mosaic pattern.

    Returns:
        (polarization_0_deg, polarization_45_deg, polarization_90_deg, polarization_135_deg)
    """
    polarization_90_deg   = image[0::2, 0::2]
    polarization_45_deg   = image[0::2, 1::2]
    polarization_135_deg  = image[1::2, 0::2]
    polarization_0_deg    = image[1::2, 1::2]
    return polarization_0_deg,polarization_45_deg,polarization_90_deg,polarization_135_deg


def separate_channels(image_path):
    """
    Read a polarization PNM image from disk, debayer it, and save 4 channel PNGs.

    Writes: <image_path>_channel_0.png, _channel_45.png, _channel_90.png, _channel_135.png
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Split channels
    channel_0,channel_45,channel_90,channel_135 = debayerPolarImage(image)

    output_file = f"{image_path}_channel_0.png"
    cv2.imwrite(output_file, channel_0)

    output_file = f"{image_path}_channel_45.png"
    cv2.imwrite(output_file, channel_45)

    output_file = f"{image_path}_channel_90.png"
    cv2.imwrite(output_file, channel_90)

    output_file = f"{image_path}_channel_135.png"
    cv2.imwrite(output_file, channel_135)

if __name__ == "__main__":
    # Check if an image file path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    separate_channels(image_path)

