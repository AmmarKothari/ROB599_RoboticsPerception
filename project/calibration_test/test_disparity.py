# a script to dive deeper into what is happening for aligning images an creating disparity




from DepthFromStero import DepthFromStereo, phonyStereoCamera, calibrateStereoCameras, horizConcat, stereoCamera
import glob
import os
import cv2
import pdb

folder = "/home/ammar/Documents/Projects/ROB599_RoboticsPerception/project/calibration_test/StereoImages/DisparityTesting/"

SC = phonyStereoCamera(os.path.join(folder, 'Pairs'), os.path.join(folder, 'Pairs'))
# SC = stereoCamera()

# load calibrated camera parameters
cal = calibrateStereoCameras()
cal.loadCheckerboardFeatures('checkerboardFeatures.npz')
cal.loadCameraParameters('CameraCalLeft.npz', 'CameraCalRight.npz')
cal.loadExtrinsicParameters('StereoCal.npz')
cal.loadRectifyParams('RectifyCal.npz')

loop_flag = True
i = 0
while loop_flag:
	left, right, loop_flag = SC.getStereoPair()
	rectL, rectR = cal.rectifyImages(left, right)
	cv2.imwrite(os.path.join(folder, 'Rectified', '%04d_left.png' %i), rectL)
	cv2.imwrite(os.path.join(folder, 'Rectified', '%04d_right.png'% i), rectR)
	i += 1