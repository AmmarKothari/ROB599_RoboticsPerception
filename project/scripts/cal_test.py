import numpy as np
import cv2
import glob
import pdb
import random, string, os


#Yi Herng Ong - last edited by 4/22/2018
#Pose estimation on images
class poseEstimation(object):
	def __init__(self, Camera_npz):
		self.fn  = Camera_npz
		self.w = 9
		self.h = 9

	def loadCameraParams(self):
		with np.load(self.fn) as X:
			self.mtx, self.dist, self.rvecs, self.tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

	def draw(self, img, corners, imgpts):
		corner = tuple(corners[0].ravel())
		img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
		img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
		img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
		return img

	def drawCube(self, img, corners, imgpts):
		imgpts = np.int32(imgpts).reshape(-1,2)

		# draw ground floor in green
		img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

		# draw pillars in blue color
		for i,j in zip(range(4),range(4,8)):
			img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

		# draw top layer in red color
		img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
		return img

	def pose_showImage(self,folder):
		self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		self.objp = np.zeros((self.w*self.h,3), np.float32)
		# if this has some meaning in space, then can get a better estimate of position
		# provide "real" location of grid points on table
		self.objp[:,:2] = np.mgrid[0:self.w,0:self.h].T.reshape(-1,2)

		self.axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
		self.axis1 = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
		img_count = 0
		for fname in glob.glob(os.path.join(folder, '*.jpg')):
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (self.w,self.h),None)
			# pdb.set_trace()

			if ret == True:
				corners2 = cv2.cornerSubPix(gray,corners,(9,9),(-1,-1),self.criteria)
				#print cv2.solvePnPRansac(self.objp, corners2, self.mtx, self.dist)[:]

				# Find the rotation and translation vectors.
				# have to associate each point found in corners with a point in objp
				# RANSAC should have some robustness in finding object so only define subset
				# define features of object to track?
				retval, rvecs, tvecs, inliers =  cv2.solvePnPRansac(self.objp, corners2, self.mtx, self.dist)[:]

				# project 3D points to image plane
				imgpts, jac = cv2.projectPoints(self.axis1, rvecs, tvecs, self.mtx, self.dist)
				pdb.set_trace()
				img = self.drawCube(img,corners2,imgpts) #choose either draw or drawCube
				cv2.imshow('img%s' %img_count,img)
				k = cv2.waitKey(1000) & 0xff
				# if k == 's':
				#     cv2.imwrite(fname[:6]+'.png', img)

				img_count += 1
				cv2.destroyAllWindows()
				pdb.set_trace()


		# cv2.destroyAllWindows() 	


if __name__ == '__main__':
	C = calibrateCameraCustom()
	img = C.loadImage('../Data/CalImages_Webcam/Image_16.jpg')
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

	#Pose Estimation
	P = poseEstimation('CameraCal.npz')
	P.loadCameraParams()
	P.pose_showImage('CalImages_Webcam')


	# pdb.set_trace()
