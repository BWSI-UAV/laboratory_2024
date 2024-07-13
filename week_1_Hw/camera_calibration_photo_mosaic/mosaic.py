import cv2
import numpy as np
import glob

Load Calibration Parameters:
    Function load_calibration(calibration_file):
        Load calibration data from the file
        Extract camera matrix and distortion coefficients
        Return camera matrix and distortion coefficients

Undistort Image:
    Function undistort_image(image, camera_matrix, dist_coeffs):
        Get image dimensions (height, width)
        Compute new camera matrix for undistortion
        Undistort the image (use cv2 undistort)
        Crop the undistorted image using ROI
        Return undistorted image

Harris Corner Detection:
    Function harris_corner_detection(image):
        Convert the image to grayscale
        Apply Harris corner detection
        Dilate corners
        Mark corners on the image
        Return image with marked corners and detected corners

Match Features Between Images:
    Function match_features(image1, image2):
        Detect keypoints and descriptors in image1 using SIFT
        Detect keypoints and descriptors in image2 using SIFT
        Match descriptors using brute-force matcher
        Extract matched points from both images
        Return matched points from image1 and image2

Create Mosaic:
    Function create_mosaic(images, camera_matrix, dist_coeffs):
        Undistort all images using undistort_image function
        Initialize mosaic with the first undistorted image
        For each subsequent undistorted image:
            Detect Harris corners in both mosaic and current image using harris_corner_detection
            Match features between mosaic and current image using match_features
            Estimate homography using matched points
            Warp mosaic image using the estimated homography
            Blend current image into mosaic
        Return final mosaic image

Main:
    Load camera matrix and distortion coefficients from calibration file
    Load images from specified directory
    Create mosaic using create_mosaic function
    Save the mosaic image to a file

# Display the mosaic image
cv2.imshow('Mosaic', mosaic_image)
cv2.waitKey(0)
cv2.destroyAllWindows()