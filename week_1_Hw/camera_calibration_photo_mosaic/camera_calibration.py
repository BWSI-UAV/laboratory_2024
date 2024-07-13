import numpy as np
import cv2
import glob

Define checkerboard_dims as (9, 6) 

Create objp as a zero array of shape (number of corners, 3), float32
Set the first two columns of objp to the coordinate grid of corners

Initialize objpoints as an empty list
Initialize imgpoints as an empty list

Load all checkerboard images using glob ('path/to/images/*.jpg')

For each image in images:
    Read the image
    Convert the image to grayscale
    
    Find the chessboard corners in the grayscale image
    If corners are found:
        Append objp to objpoints
        Refine corner positions using cornerSubPix
        Append refined corners to imgpoints
        
        Optionally, draw chessboard corners on the image
        Optionally, display the image with drawn corners
        Wait for a short period
    
Destroy all OpenCV windows

Calibrate the camera using calibrateCamera with objpoints, imgpoints, and image size
Get the camera matrix, distortion coefficients, rotation vectors, and translation vectors

Save the calibration results (camera matrix, distortion coefficients) to a file. 
A common and convenient format for storing camera calibration data is the NumPy .npz file format,
    which allows you to store multiple NumPy arrays in a single compressed file.

Verify the calibration:
    Initialize mean_error to 0
    For each pair of object points and image points:
        Project the object points to image points using projectPoints
        Compute the error between the projected and actual image points
        Accumulate the error
    Compute the average error
    Print the total average error
