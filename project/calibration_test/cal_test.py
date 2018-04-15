import numpy as np
import cv2
import glob
import pdb
import random, string, os


def horizConcat(img1, img2):
	y = min(img1.shape[0], img2.shape[0])
	x = min(img1.shape[1], img2.shape[1])
	c = min(img1.shape[2], img2.shape[2])
	vis = np.concatenate((img1[:y,:,:c], img2[:y,:,:c]), axis=1)
	return vis

def avgImage(img1, img2):
	y = min(img1.shape[0], img2.shape[0])
	x = min(img1.shape[1], img2.shape[1])
	c = min(img1.shape[2], img2.shape[2])
	img_avg = 0.5*(img1[:y,:x,:c]+img2[:y,:x,:c])
	return img_avg


class calibrateCameraCustom(object):
	def __init__(self):
		i = 1
		# number of internal corners
		self.w = 9
		self.h = 9
		self.objpoints = [] # 3d point in real world space
		self.imgpoints = [] # 2d points in image plane.

	def loadImages(self, folder='.'):
		# pass the folder that has the calibration images
		images = glob.glob(os.path.join(folder, '*.jpg'))
		for fname in images:
			img = self.loadImage(fname)
			gray = self.grayscaleImage(img)
			thresh = self.thresholdImage(gray)
			ret, corners = self.findChessboardCorners(thresh)
			if ret:
				chess, opts, imgpts = self.drawChessboardCorners(corners, gray, ret)
				self.getCameraCalibrationParameters(gray, opts, imgpts)
		img2 = self.loadImage(fname)
		self.optimalNewCameraMatrix(img2)
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

	def getCameraCalibrationParameters(self, gray, objpoints, imgpoints):
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
		self.ret = ret
		self.mtx = mtx
		self.dist = dist
		self.rvecs = rvecs
		self.tvecs = tvecs

	def optimalNewCameraMatrix(self, img):
		h,  w = img.shape[:2]
		newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
		self.newcameramtx = newcameramtx
		self.roi = roi

	def getCameraParams(self):
		return [self.ret, self.mtx, self.dist, self.rvecs, self.tvecs, self.newcameramtx, self.roi]

	def saveCameraParams(self, folder):
		fn = os.path.join(folder,'CameraCal')
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
		print "total error: ", mean_error/len(self.objpoints)
    
class undistortImage(object):
	def __init__(self, ret, mtx, dist, rvecs, tvecs, newcameramtx, roi):
		# set all parameters here that can be used to undistort
		self.ret = ret
		self.mtx = mtx
		self.dist = dist
		self.rvecs = rvecs
		self.tvecs = tvecs
		self.newcameramtx = newcameramtx
		self.roi = roi

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
		return dst

	def showImage(self, img):
		title = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
		cv2.imshow(title,img)
		cv2.waitKey(0)





if __name__ == '__main__':
	C = calibrateCameraCustom()
	# img = C.loadImage()
	# gray = C.grayscaleImage(img)
	# thresh = C.thresholdImage(gray)
	# ret, corners = C.findChessboardCorners(thresh)
	# if ret:
	# 	chess, opts, imgpts = C.drawChessboardCorners(corners, gray)
	# 	C.getCameraCalibrationParameters(gray, opts, imgpts)
	# 	img2 = C.loadImage('Image_1.jpg')
	# 	C.optimalNewCameraMatrix(img2)
	# 	C.reProjectionError()
		# C.showImage(chess)
	# C.showImage(thresh)
	# C.showImage(gray)
	# C.loadImages('CalImages_Webcam')
	C.loadCameraParams('CalImages_Webcam/CameraCal.npz')
	U = undistortImage(*C.getCameraParams())
	undist = U.undistortRectify(img)
	# U.showImage(horizConcat(undist, img))
	# U.showImage(avgImage(undist,img))
	pdb.set_trace()
