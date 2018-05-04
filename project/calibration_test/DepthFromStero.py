'''
class to take in depth images
outputs a disparity map

'''

import cv2
import numpy as np
import pdb
from cal_test import calibrateCameraCustom, horizConcat
import glob, os, time, subprocess
import roslaunch
import cv_bridge
import rospy
import sensor_msgs.msg
import copy

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
		left_img, right_img, __ = self._getCurrentStereoPair()
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
		y,x,c = left_img.shape
		crop_border = 10
		left_img = left_img[crop_border:y-crop_border, crop_border:x-crop_border, :]
		right_img = right_img[crop_border:y-crop_border, crop_border:x-crop_border, :]
		# pdb.set_trace()
		return left_img, right_img, True

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

class stereoCamera(object):
	def __init__(self):
		self.camL = cv2.VideoCapture()
		self.camR = cv2.VideoCapture()

		self.crop_flag = True

		available_cams = self.getCameraIndexes()
		if len(available_cams) >= 2:
			retL, retR = self.openCameras(available_cams)
		else:
			print("Could not open Cameras")
			raise ValueError('Could not open Cameras')

		if not retL or not retR:
			print("Could not open Cameras")
			raise ValueError('Could not open Cameras')

		CAMERA_WIDTH = 1280
		CAMERA_HEIGHT = 720
		self.camL.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
		self.camL.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
		self.camR.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
		self.camR.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
		self.camL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
		self.camR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

		subprocess.Popen(['v4l2-ctl', '-c', 'focus_auto=0', '-d', '/dev/video%s' %available_cams[0]])
		subprocess.Popen(['v4l2-ctl', '-c', 'focus_auto=0', '-d', '/dev/video%s' %available_cams[1]])
		subprocess.Popen(['uvcdynctrl'. '-v', '-d', 'video%s' %available_cams[0], '--set=Focus, Auto', '0'])
		subprocess.Popen(['uvcdynctrl'. '-v', '-d', 'video%s' %available_cams[1], '--set=Focus, Auto', '0'])

		self.CROPPED = [960, 720]

	def openCameras(self, cams_idxs):
		retL = self.camL.open(cams_idxs[1])
		retR = self.camR.open(cams_idxs[0])
		return retL, retR

	def getCameraIndexes(self):
		cam = cv2.VideoCapture()
		available_cameras = []
		for i in range(10):
			if cam.open(i):
				available_cameras.append(i)
		print(available_cameras)
		return available_cameras

	def getStereoPair(self):
		# returns a stereo pair and a flag saying if images remaining
		left_img, right_img, flag = self._getCurrentStereoPair()
		return left_img, right_img, flag

	def _getCurrentStereoPair(self):
		self.camL.grab()
		time.sleep(0.1)
		self.camR.grab()
		retL, left_img = self.camL.retrieve();
		retR, right_img = self.camR.retrieve();
		ret = retL and retR
		if self.crop_flag:
			left_img, right_img = self.cropImage(self.CROPPED[0], self.CROPPED[1], left_img, right_img)
		return left_img, right_img, ret

	def cropImage(self, y_size, x_size, left, right):
		dims_l = left.shape
		dims_r = right.shape
		y_start = 0.5*(dims_l[0]-y_size)
		y_end = dims_l[0]-0.5*(dims_l[0]-y_size)
		x_start = 0.5*(dims_l[1]-x_size)
		x_end = dims_l[1]-0.5*(dims_l[1]-x_size)
		left_crop = left[y_start:y_end,x_start:x_end,:]
		right_crop = right[y_start:y_end,x_start:x_end,:]
		return left_crop, right_crop

	def stream(self):
		# stream images
		keep_streaming = True
		while keep_streaming:
			left, right, __ = self.getStereoPair()
			cv2.imshow('Streaming', np.hstack((left,right)));

			key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
			if (key == ord('q')):
				keep_streaming = False;

	def recordDataStream(self, n_frames=100, t_pause=0.1, folder = '.'):
		# records n_frames
		# pauses t_pause between each frame
		# saves images to folder

		for i in range(n_frames):
			left, right, __ = self.getStereoPair()
			cv2.imshow('Recording', np.hstack((left,right)));
			key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
			cv2.imwrite(os.path.join(folder, '%04d_left.png' %i), left)
			cv2.imwrite(os.path.join(folder, '%04d_right.png' %i), right)
			time.sleep(t_pause)

		cv2.destroyAllWindows()

topics = ['/left/image_raw', '/right/image_raw']
class rosStereoCamera(object):
	# camera object that relies on ROS interface
	# kept running into no space left on device issue
	def __init__(self):
		self.launchStereoNode()
		self.bridge = cv_bridge.CvBridge()

	def launchStereoNode(self):
		# turns off autofocus -- make sure these are the right cameras
		subprocess.Popen(['v4l2-ctl', '-c', 'focus_auto=0', '-d', '/dev/video0'])
		subprocess.Popen(['v4l2-ctl', '-c', 'focus_auto=0', '-d', '/dev/video1'])
		package = 'uvc_camera'
		executable = 'uvc_stereo_node'
		self.node = roslaunch.core.Node(package, executable)
		self.launch = roslaunch.scriptapi.ROSLaunch()
		self.launch.start()
		self.process = self.launch.launch(self.node)

	def closeStereoNode():
		self.process.stop()

	def getMessages(self, topics):
		d = dict.fromkeys(topics)
		for t in topics:
			d[t] = rospy.wait_for_message(t, sensor_msgs.msg.Image, timeout = 1)
		return d

	def getStereoPair(self):
		img_dict = self.getMessages(topics)
		left_img = img_dict[topics[0]]
		right_img = img_dict[topics[1]]
		flag = True
		if left_img is None or right_img is None:
			flag = False
		return left_img, right_img, flag






windowNameL = "LEFT Camera Input"; # window name
windowNameR = "RIGHT Camera Input"; # window name
WIDTH_SQUARE = 4
HEIGHT_SQUARES = 3
SQUARE_SIZE_MM = 51.5

WIDTH_SQUARE = 7
HEIGHT_SQUARES = 9
SQUARE_SIZE_MM = 22

FOLDER = '/home/ammar/Documents/Projects/ROB599_RoboticsPerception/project/calibration_test/StereoImages/Calibration'

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

	def leftThresholdImage(self, img, gray_val = 70):
		ret,thresh1 = cv2.threshold(img,gray_val,255,cv2.THRESH_BINARY)
		# pdb.set_trace()
		return thresh1

	def rightThresholdImage(self, img, gray_val = 60):
		ret,thresh1 = cv2.threshold(img,gray_val,255,cv2.THRESH_BINARY)
		# thresh1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
		# pdb.set_trace()
		return thresh1

	def extractCheckerboardFeatures(self, stereo_camera):
		chessboard_criteria = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
		termination_criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		detection_dict = dict.fromkeys(['Left', 'Right'], np.zeros(3))
		# idx 0: chessboard_pattern_detections
		# idx 1: last_chessboard_pattern_detections
		# idx 2: failed_chessboard_pattern_detections
		chessboard_pattern_detections = 0
		loop_flag = True
		show_flag = True
		while loop_flag:
			left_img, right_img, loop_flag = stereo_camera.getStereoPair()

			# could make this iterative so that it tries different thresholding
			# or could write a manual corner finder
			left_gray, right_gray = self.grayscaleImage(left_img), self.grayscaleImage(right_img)
			for thresh in range(50,150,10):
				left_thresh = self.leftThresholdImage(left_gray, gray_val = thresh)
				retR, cornersL = cv2.findChessboardCorners(left_thresh, (self.w,self.h),None, chessboard_criteria)
				if (retR == True):
					break
			for thresh in range(50,150,10):
				right_thresh = self.rightThresholdImage(right_gray, gray_val = thresh)
				retL, cornersR = cv2.findChessboardCorners(right_thresh, (self.w,self.h),None, chessboard_criteria)
				if (retL == True):
					break

			# if retR:
			# 	detection_dict['Right'][0] += 1
			# 	detection_dict['Right'][2] = 0
			# else:
			# 	detection_dict['Right'][2] += 1
			# if retL: detection_dict['Left'][0] += 1
			print("Left Corners: {}, Right Corners: {}".format(retR, retL))
			# pdb.set_trace()

			if ((retR == True) and (retL == True)):
				cv2.imwrite(os.path.join(FOLDER, 'RGBPairs', '%04d_left.png' %chessboard_pattern_detections), left_img)
				cv2.imwrite(os.path.join(FOLDER, 'RGBPairs', '%04d_right.png' %chessboard_pattern_detections), right_img)
				cv2.imwrite(os.path.join(FOLDER, 'ThreshPairs', '%04d_left.png' %chessboard_pattern_detections), left_thresh)
				cv2.imwrite(os.path.join(FOLDER, 'ThreshPairs', '%04d_right.png' %chessboard_pattern_detections), right_thresh)
				chessboard_pattern_detections += 1
				if chessboard_pattern_detections > 50:
					# stop looking for checkerboard images when we get to 25 sets
					loop_flag = False

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

				cv2.imwrite(os.path.join(FOLDER, 'DrawBoardPairs', '%04d_left.png' %chessboard_pattern_detections), drawboardL)
				cv2.imwrite(os.path.join(FOLDER, 'DrawBoardPairs', '%04d_right.png' %chessboard_pattern_detections), drawboardR)

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

			key = cv2.waitKey(10) & 0xFF; # wait 500ms between frames
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

	def calibrateStereoCameras(self, img_gray): #intrinsic calibration
		# can this be done with sets of images?
		# average over the set of images
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

	def loadCameraParameters(self, fn_left, fn_right):
		# load camera parameters
		self.loadCameraParams(fn_left)
		self.retL, self.mtxL, self.distL, self.rvecsL, self.tvecsL, self.newcameramtxL, self.roiL = self.getCameraParams()
		self.loadCameraParams(fn_right)
		self.retR, self.mtxR, self.distR, self.rvecsR, self.tvecsR, self.newcameramtxR, self.roiR = self.getCameraParams()

	def extrinsicCalibrateStereoCameras(self, img_gray):
		# recover the relative camera pose
		termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
		(rms_stereo, self.camera_matrix_l, self.dist_coeffs_l, self.camera_matrix_r, self.dist_coeffs_r, self.R, self.T, E, F) = \
			cv2.stereoCalibrate(self.objpoints, self.imgpointsL, self.imgpointsR, self.mtxL, self.distL, self.mtxR, self.distR,  img_gray.shape[::-1], criteria=termination_criteria_extrinsics, flags=0);
		print("STEREO: RMS left to  right re-projection error: ", rms_stereo)
		print("Good results are between 0.1 and 1.0, but no real guidelines.")
		self.saveExtrinsicParameters('StereoCal')

	def saveExtrinsicParameters(self, fn):
		np.savez(fn, camera_matrix_l=self.camera_matrix_l, dist_coeffs_l=self.dist_coeffs_l, camera_matrix_r=self.camera_matrix_r, dist_coeffs_r=self.dist_coeffs_r, R=self.R, T=self.T)

	def loadExtrinsicParameters(self, fn):
		npzfile = np.load(fn)
		self.camera_matrix_l = npzfile['camera_matrix_l']
		self.dist_coeffs_l = npzfile['dist_coeffs_l']
		self.camera_matrix_r = npzfile['camera_matrix_r']
		self.dist_coeffs_r = npzfile['dist_coeffs_r']
		self.R = npzfile['R']
		self.T = npzfile['T']

	def rectifyCalibrate(self, img_gray):
		# align left and right images
		# flags Operation flags that may be zero or CV_CALIB_ZERO_DISPARITY . If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area.
		self.RL, self.RR, self.PL, self.PR, self.Q, _, _ = cv2.stereoRectify(self.camera_matrix_l, self.dist_coeffs_l, self.camera_matrix_r, self.dist_coeffs_r,  img_gray.shape[::-1], self.R, self.T, alpha=1, flags=1)
		# self.mapL1, self.mapL2 = cv2.initUndistortRectifyMap(self.camera_matrix_l, self.dist_coeffs_l, self.RL, self.PL, img_gray.shape[::-1], cv2.CV_32FC1)
		# self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(self.camera_matrix_r, self.dist_coeffs_r, self.RR, self.PR, img_gray.shape[::-1], cv2.CV_32FC1)
		self.mapL1, self.mapL2 = cv2.initUndistortRectifyMap(self.camera_matrix_l, self.dist_coeffs_l, self.RL, self.PL, img_gray.shape[::-1], cv2.CV_32FC1)
		self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(self.camera_matrix_r, self.dist_coeffs_r, self.RR, self.PR, img_gray.shape[::-1], cv2.CV_32FC1)
		self.saveRectifyParams('RectifyCal')
		print("P Left: {}".format(self.PL))
		print("P Right: {}".format(self.PR))

	def saveRectifyParams(self, fn):
		np.savez(fn, RL=self.RL, RR=self.RR, PL=self.PL, PR=self.PR, mapL1=self.mapL1, mapL2=self.mapL2, mapR1=self.mapR1, mapR2=self.mapR2)

	def loadRectifyParams(self, fn):
		npzfile = np.load(fn)
		self.RL = npzfile['RL']
		self.RR = npzfile['RR']
		self.PL = npzfile['PL']
		self.PR = npzfile['PR']
		self.mapL1 = npzfile['mapL1']
		self.mapL2 = npzfile['mapL2']
		self.mapR1 = npzfile['mapR1']
		self.mapR2 = npzfile['mapR2']

	def rectifyImages(self, left_img, right_img):
		undistorted_rectifiedL = cv2.remap(left_img, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
		undistorted_rectifiedR = cv2.remap(right_img, self.mapR1, self.mapR2, cv2.INTER_LINEAR)
		return undistorted_rectifiedL, undistorted_rectifiedR

	def disparityImage(self, left_img, right_img):
		max_disparity = 16;
		block_size = 15
		max_disparity = 128;
		block_size = 15
		stereoProcessor = cv2.StereoSGBM_create(1, max_disparity, block_size);
		left_gray = cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY);
		right_gray = cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY);

		left_rect, right_rect = self.rectifyImages(left_gray, right_gray)
		# compute disparity image from undistorted and rectified versions
		# (which for reasons best known to the OpenCV developers is returned scaled by 16)

		disparity = stereoProcessor.compute(right_rect, left_rect);
		orig_disparity = copy.deepcopy(disparity)
		# pdb.set_trace()
		cv2.filterSpeckles(disparity, 0, 4000, 128);
		# cv2.filterSpeckles(disparity, 0, 400, 64);

		# scale the disparity to 8-bit for viewing
		disparity_scaled = (disparity / 16.).astype(np.uint8) + abs(disparity.min())

		# is the unscaled version more useful for detection or something?
		return disparity_scaled






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
