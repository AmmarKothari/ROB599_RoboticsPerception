import numpy as np
import cv2
import glob
import pdb
import random, string

# termination criteria
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.

# images = glob.glob('*.jpg')
# # images = glob.glob('*.png')
# pdb.set_trace()
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)

#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         imgpoints.append(corners2)

#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
#         cv2.imshow('img',img)
#         cv2.waitKey(500)

# cv2.destroyAllWindows()


class calibrateCameraCustom(object):
	def __init__(self):
		i = 1
		# number of internal corners
		self.w = 9
		self.h = 9

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

	def drawChessboardCorners(self, corners, gray):
		objp = np.zeros((self.w*self.h,3), np.float32)
		objp[:,:2] = np.mgrid[0:self.w,0:self.h].T.reshape(-1,2)
		self.objpoints = []
		self.objpoints.append(objp) # 3d point in real world space
		self.imgpoints = [] # 2d points in image plane.
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),criteria)
		self.imgpoints.append(corners2)
		# Draw and display the corners
		img = cv2.drawChessboardCorners(gray, (self.w,self.h), corners2,ret)
		return img, self.objpoints, self.imgpoints

	def getCameraCalibrationParameters(self, gray, objpoints, imgpoints):
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
		self.ret = ret
		self.mtx = mtx
		self.dist = dist
		self.rvecs = rvecs
		self.tvecs = tvecs
		pdb.set_trace()
    





if __name__ == '__main__':
	C = calibrateCameraCustom()
	img = C.loadImage()
	gray = C.grayscaleImage(img)
	thresh = C.thresholdImage(gray)
	ret, corners = C.findChessboardCorners(thresh)
	if ret:
		chess, opts, imgpts = C.drawChessboardCorners(corners, gray)
		C.getCameraCalibrationParameters(gray, opts, imgpts)
		# C.showImage(chess)
	# C.showImage(thresh)
	# C.showImage(gray)
	# pdb.set_trace()
