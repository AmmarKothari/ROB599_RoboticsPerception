'''
class to take in depth images
outputs a disparity map

'''

import cv2
import numpy as np
import pdb

class DepthFromStereo():
	def __init__(self):
		i = 1

	def calibrateSubPartOfImage(self):
		termination_criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


	def combineImagesIntoDisparity(self, left_img, right_img):
		# disparity settings
		window_size = 5
		min_disp = 32
		num_disp = 112-min_disp
		stereo = cv2.StereoSGBM_create(
			minDisparity = min_disp,
			numDisparities = num_disp,
			# SADWindowSize = window_size,
			uniquenessRatio = 10,
			speckleWindowSize = 100,
			speckleRange = 32,
			disp12MaxDiff = 1,
			P1 = 8*3*window_size**2,
			P2 = 32*3*window_size**2,
			# fullDP = False
		)
		 
		# morphology settings
		kernel = np.ones((12,12),np.uint8)
	 
		# compute disparity
		disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
		disparity = (disparity-min_disp)/num_disp

		return disparity

	def showImg(self, img):
		cv2.imshow('idontknow', img)
		cv2.waitKey(0)
