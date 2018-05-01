'''
class to take in depth images
outputs a disparity map

'''

import cv2
import numpy as np
import pdb
from cal_test import calibrateCameraCustom
import glob, os

class phonyStereoCamera(object):
	# creates an interface for a fake camera sort of like a real camera
	def __init__(self, left_folder, right_folder):
		self.left_folder, self.right_folder = left_folder, right_folder
		self.left_fns, self.right_fns = self.getImagePathsFromFolder(left_folder, 'left'), self.getImagePathsFromFolder(right_folder, 'right')
		self.i = 0
		self.total_frames = len(self.left_fns)


	def getImagePathsFromFolder(self, folder, side):
		# return a list of files in folder path
		fnames = [f for f in glob.glob(os.path.join(folder, '*.png')) if side in f]
		fnames.sort()
		return fnames

	def getStereoPair(self):
		# returns a stereo pair and a flag saying if images remaining
		left_img, right_img = self._getCurrentStereoPair()
		self.i += 1
		flag = True

		if self.total_frames<=self.i:
			self._resetFrameCount()
			flag = False

		return left_img, right_img, flag

	def _getCurrentStereoPair(self, i=None):
		if i is None:
			i = self.i
		left_img, right_img = self.loadImage(self.left_fns[i]), self.loadImage(self.right_fns[i])
		return left_img, right_img

	def _getCurrentFileNames(self):
		return self.left_fns[self.i], self.right_fns[self.i]

	def _resetFrameCount(self):
		self.i=0

	def _printPairNames(self):
		# display the names of files that are paired up
		for lf,rf in zip(self.left_fns, self.right_fns):
			print("Left Image: {}, Right Image: {}".format(lf, rf))

	def loadImage(self, fname):
		img = cv2.imread(fname)
		return img



windowNameL = "LEFT Camera Input"; # window name
windowNameR = "RIGHT Camera Input"; # window name
WIDTH_SQUARE = 4
HEIGHT_SQUARES = 3
SQUARE_SIZE_MM = 40

class calibrateStereoCameras(calibrateCameraCustom):
	def __init__(self):
		super(calibrateStereoCameras, self).__init__()
		self.w = WIDTH_SQUARE
		self.h = HEIGHT_SQUARES
		# self.w = HEIGHT_SQUARES
		# self.h = WIDTH_SQUARE
		self.square_size_mm = SQUARE_SIZE_MM

		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

		objp = np.zeros((self.w*self.h,3), np.float32)
		objp[:,:2] = np.mgrid[0:self.w,0:self.h].T.reshape(-1,2)
		self.objp = objp * self.square_size_mm;

		self.objpoints = [] # 3d point in real world space
		self.imgpointsR = [] # 2d points in image plane.
		self.imgpointsL = [] # 2d points in image plane.

	def leftThresholdImage(self, img):
		ret,thresh1 = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
		# pdb.set_trace()
		return thresh1

	def rightThresholdImage(self, img):
		ret,thresh1 = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
		# thresh1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
		# pdb.set_trace()
		return thresh1

	def extractCheckerboardFeatures(self, stereo_camera):
		chessboard_criteria = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
		termination_criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		chessboard_pattern_detections = 0
		loop_flag = True
		show_flag = False
		while loop_flag:
			left_img, right_img, loop_flag = stereo_camera.getStereoPair()

			# could make this iterative so that it tries different thresholding
			# or could write a manual corner finder
			left_gray, right_gray = self.grayscaleImage(left_img), self.grayscaleImage(right_img)
			left_thresh, right_thresh = self.leftThresholdImage(left_gray), self.rightThresholdImage(right_gray)

			retR, cornersL = cv2.findChessboardCorners(left_thresh, (self.w,self.h),None, chessboard_criteria)
			retL, cornersR = cv2.findChessboardCorners(right_thresh, (self.w,self.h),None, chessboard_criteria)
			print("Left Corners: {}, Right Corners: {}".format(retR, retL))
			# pdb.set_trace()

			if ((retR == True) and (retL == True)):
				# pdb.set_trace()
				chessboard_pattern_detections += 1

				# add object points to global list

				self.objpoints.append(self.objp)

				# refine corner locations to sub-pixel accuracy and then

				corners_sp_L = cv2.cornerSubPix(left_thresh,cornersL,(11,11),(-1,-1),termination_criteria_subpix);
				self.imgpointsL.append(corners_sp_L);
				corners_sp_R = cv2.cornerSubPix(right_thresh,cornersR,(11,11),(-1,-1),termination_criteria_subpix);
				self.imgpointsR.append(corners_sp_R);

				# Draw and display the corners

				drawboardL = cv2.drawChessboardCorners(left_img, (self.w,self.h), corners_sp_L,retL)
				drawboardR = cv2.drawChessboardCorners(right_img, (self.w,self.h), corners_sp_R,retR)

				if show_flag:
					text = 'detected: ' + str(chessboard_pattern_detections);
					cv2.putText(drawboardL, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8);
					cv2.imshow(windowNameL,drawboardL);
					cv2.imshow(windowNameR,drawboardR);
			else:
				if show_flag:
					text = 'detected: ' + str(chessboard_pattern_detections);
					cv2.putText(left_thresh, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8);
					cv2.imshow(windowNameL,left_thresh);
					cv2.imshow(windowNameR,right_thresh);

			# key = cv2.waitKey(100) & 0xFF; # wait 500ms between frames
		print('Images with Checkerboard Detection: {}'.format(chessboard_pattern_detections))
		if len(self.objpoints) == 0 or len(self.imgpointsL) == 0 or len(self.imgpointsR) == 0:
			print('Error no features found!')
			return -1
		np.savez('checkerboardFeatures', objpoints = self.objpoints, imgpointsL=self.imgpointsL, imgpointsR=self.imgpointsR)

	def loadCheckerboardFeatures(self, fn):
		npzfile = np.load(fn)
		self.objpoints = npzfile['objpoints']
		self.imgpointsL = npzfile['imgpointsL']
		self.imgpointsR = npzfile['imgpointsR']

	def reProjectionError(self, imgpoints):
		# just setting a class value before calling function
		self.imgpoints = imgpoints
		super(calibrateStereoCameras, self).reProjectionError()

	def calibrateStereoCameras(self, img_gray):
		# Left Camera
		self.getCameraCalibrationParameters(img_gray, self.objpoints, self.imgpointsL)
		self.optimalNewCameraMatrix(img_gray)
		ret, mtxL, distL, rvecsL, tvecsL, newcameramtxL, roiL = self.getCameraParams()
		self.reProjectionError(self.imgpointsL)
		self.saveCameraParams('.', 'CameraCalLeft')

		# Right Camera
		self.getCameraCalibrationParameters(img_gray, self.objpoints, self.imgpointsR)
		self.optimalNewCameraMatrix(img_gray)
		ret, mtxR, distR, rvecsR, tvecsR, newcameramtxR, roiR = self.getCameraParams()
		self.reProjectionError(self.imgpointsR)
		self.saveCameraParams('.', 'CameraCalRight')

	def loadCameraParameters(self, fn_left, fn_right): #### start here!!!!
		# load camera parameters
		self.loadCameraParams(fn_left)
		self.mtxL, self.distL, self.rvecsL, self.tvecsL, self.newcameramtxL, self.roiL = self.getCameraParams()




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
