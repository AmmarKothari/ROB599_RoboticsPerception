### Calibrate camera code


import numpy as np
import cv2
import glob
import pdb
import random, string, os


class calibrateCameraCustom(object):
	def __init__(self, camera, w=9, h=9):
		# camera is a openCV camera object
		# number of internal corners
		self.w = w
		self.h = h
		self.img_w = 0
		self.img_h = 0
		self.objpoints = [] # 3d point in real world space
		self.imgpoints = [] # 2d points in image plane.
		self.images = [] # list of all images that are good

	def loadImages(self, folder='.'):
		# pass the folder that has the calibration images
		images = glob.glob(os.path.join(folder, '*.jpg'))
		for fname in images:
			img = self.loadImage(fname)
			gray = self.grayscaleImage(img)
			thresh = self.thresholdImage(gray)
			ret, corners = self.findChessboardCorners(gray)
			if ret:
				chess, opts, imgpts = self.drawChessboardCorners(corners, gray, ret)
				self.getCameraCalibrationParameters(gray, opts, imgpts)
		img2 = self.loadImage(fname)
		self.optimalNewCameraMatrix(img2)
		self.saveCameraParams(folder)
		self.reProjectionError()

	def addImageToCal(self, image):
		# add image to the calibration set
		gray = self.grayscaleImage(image)
		ret, corners = self.findChessboardCorners(gray)
		self.img_w = gray.shape[1]
		self.img_h = gray.shape[0]
		if ret:
			chess, opts, imgpts = self.drawChessboardCorners(corners, gray, ret)
			self.images.append(image)
		return ret

	def cal(self, folder = '.'):
		# calibrate over all images in the set
		self.getCameraCalibrationParameters(self.objpoints, self.imgpoints)
		self.optimalNewCameraMatrix()
		self.saveCameraParams(folder)
		self.reProjectionError()



	def loadImage(self, fname='Image_0.jpg'):
		img = cv2.imread(fname)
		return img

	def grayscaleImage(self, img):
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		return gray

	def showImage(self, img):
		title = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
		cv2.imshow(title,img)
		cv2.waitKey(0)

	def thresholdImage(self, img):
		ret,thresh1 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
		return thresh1

	def findChessboardCorners(self, gray):
		ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h),None)
		return ret, corners

	def drawChessboardCorners(self, corners, gray, ret):
		objp = np.zeros((self.w*self.h,3), np.float32)
		objp[:,:2] = np.mgrid[0:self.w,0:self.h].T.reshape(-1,2)
		self.objpoints.append(objp) 
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		self.imgpoints.append(corners2)
		# Draw and display the corners
		img = cv2.drawChessboardCorners(gray, (self.w,self.h), corners2,ret)
		return img, self.objpoints, self.imgpoints

	def getCameraCalibrationParameters(self, objpoints, imgpoints):
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (self.img_w, self.img_h),None,None)
		self.ret = ret
		self.mtx = mtx
		self.dist = dist
		self.rvecs = rvecs
		self.tvecs = tvecs

	def optimalNewCameraMatrix(self, img=None):
		if img is None:
			h,w = self.img_h, self.img_w
		else:
			h,  w = img.shape[:2]
		newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
		self.newcameramtx = newcameramtx
		self.roi = roi

	def getCameraParams(self):
		return [self.ret, self.mtx, self.dist, self.rvecs, self.tvecs, self.newcameramtx, self.roi]

	def saveCameraParams(self, folder, fn=None):
		if fn is None:
			fn = 'CameraCal'
		fn = os.path.join(folder, fn)

		np.savez(fn, ret=self.ret, mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs, newcameramtx=self.newcameramtx, roi=self.roi)

	def loadCameraParams(self, fn):
		npzfile = np.load(fn)
		print(npzfile.files)
		self.ret = npzfile['ret']
		self.mtx = npzfile['mtx']
		self.dist = npzfile['dist']
		self.rvecs = npzfile['rvecs']
		self.tvecs = npzfile['tvecs']
		self.newcameramtx = npzfile['newcameramtx']
		self.roi = npzfile['roi']

	def setCameraParams(self, cameraParams):
		self.ret, self.mtx, self.dist, self.rvecs, self.tvecs, self.newcameramtx, self.roi = cameraParams

	def reProjectionError(self):
		mean_error = 0
		tot_error = 0
		for i in xrange(len(self.objpoints)):
			imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
			error = cv2.norm(self.imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
			tot_error += error
		print "total error: ", tot_error/len(self.objpoints)


class undistortImage(object):
	def __init__(self, ret, mtx=0, dist=0, rvecs=0, tvecs=0, newcameramtx=0, roi=0):
		# set all parameters here that can be used to undistort
		if type(ret) == int:
			self.ret = ret
			self.mtx = mtx
			self.dist = dist
			self.rvecs = rvecs
			self.tvecs = tvecs
			self.newcameramtx = newcameramtx
			self.roi = roi
		else:
			# cal object
			self.ret = ret.ret
			self.mtx = ret.mtx
			self.dist = ret.dist
			self.rvecs = ret.rvecs
			self.tvecs = ret.tvecs
			self.newcameramtx = ret.newcameramtx
			self.roi = ret.roi


	def undistort(self, img):
		dst = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
		x,y,w,h = self.roi
		dst = dst[y:y+h, x:x+w]
		return dst

	def undistortRectify(self, img):
		h,  w = img.shape[:2]
		# undistort
		mapx,mapy = cv2.initUndistortRectifyMap(self.mtx,self.dist,None,self.newcameramtx,(w,h),5)
		dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

		# crop the image
		x,y,w,h = self.roi
		dst = dst[y:y+h, x:x+w]

		if not any([i == 0 for i in dst.shape]):
			success = True
		else:
			success = False


		return dst, success

	def showImage(self, img):
		title = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
		cv2.imshow(title,img)
		cv2.waitKey(0)


