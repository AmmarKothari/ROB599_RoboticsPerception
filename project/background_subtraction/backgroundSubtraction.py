
'''
remove background from a video with a base frame that is the background

'''


import cv2
import numpy as np
import pdb
# from cal_test import calibrateCameraCustom, horizConcat
import glob, os, time, subprocess
import roslaunch
import cv_bridge
import rospy
import sensor_msgs.msg
import copy
import time



def showImage(img):
	cv2.imshow('Frame', img);
	show_flag = True
	t_start = time.time()
	while show_flag:
		if cv2.waitKey(100) & 0xFF == ord('q'):
			break
		if time.time() - t_start > 100:
			break
	cv2.destroyAllWindows()


class removeBackground(object):
	def __init__(self, path_to_background):
		self.path_to_background = path_to_background
		self.background = cv2.imread(self.path_to_background)
		self.background_gray = cv2.cvtColor(self.background,cv2.COLOR_BGR2GRAY)

	def removeBackground(self, img):
		# pdb.set_trace()
		img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		diff_gray = cv2.absdiff(img_gray, self.background_gray)
		ret,thresh = cv2.threshold(diff_gray,60,255,cv2.THRESH_BINARY)
		return ret, thresh

	def maskBackground(self, img):
		ret, thresh = self.removeBackground(img)
		img_fg = cv2.bitwise_and(img, img,mask = thresh)
		return True, img_fg


# class cvRemoveBackground(object):
# 	def __init__(self):
# 		self.fgbg = cv2.createBackgroundSubtractorMOG2()

class objectDetection(object):
	# find an object in the entire image
	# return approximate location
	def __init__(self):
		# , object_only_img
		i = 1
		

	def estimatePosition(self, img):
		# given an image, estimates the position of the object
		# useful if there is no object in the frame
		if len(img.shape) == 3:
			img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		else:
			img_gray = img

		#makes it easier to detect contours -- may not be great for small objects
		# erode and then dilate may help
		img_blur = cv2.medianBlur(img_gray, 11)
		img2, contours, hierarchy = cv2.findContours(img_blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) == 0:
			# object may not be in image!
			return [-1, -1], []
		object_cnt_idx = self.findLargestContour(contours)
		[x,y] = self.getContourCentroid(contours[object_cnt_idx])
		return [x,y], contours[object_cnt_idx]

	def findLargestContour(self, contours):
		return np.argmax([cv2.contourArea(cnt) for cnt in contours])

	def getContourCentroid(self, cnt):
		M = cv2.moments(cnt)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		return [cx, cy]

	def getBoundingRect(self, cnt):
		y,x,h,w = cv2.boundingRect(cnt)
		return y,x,y+h,x+w


class cvObjectTracking(object):
	# use the built in tracking for openCV
	def __init__(self, bbox = None, img = None):
		# self.tracker = cv2.TrackerMIL_create()
		self.tracker = cv2.TrackerKCF_create()
		self.bbox = bbox
		if self.bbox is not None and img is not None:
			try:
				img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			except:
				pass
			self.tracker.init(img, self.bbox)


	def selectBoundingBox(self, img):
		roi = cv2.selectROI("tracker", img)
		self.bbox = roi

	def trackObject(self, img):
		# if type(self.tracker) == cv2.TrackerKCF:
		# 	pdb.set_trace()
		# 	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, bbox = self.tracker.update(img)
		return ret, bbox

	def drawBoundingBox(self, img, bbox):
		bbox = [int(b) for b in bbox]
		img_box = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,200,100), thickness = 5)
		return img_box









if __name__ == '__main__':
	# camera = cv2.VideoCapture(0)
	camera = cv2.VideoCapture('background_subtraction.avi')
	font = cv2.FONT_HERSHEY_SIMPLEX
	# ret, frame = camera.read(); cv2.imshow('frame', frame); cv2.waitKey(1000); cv2.destroyAllWindows()
	# cv2.imwrite('background.png', frame)




	# background = cv2.imread('Background_only.png')
	# img1 = cv2.imread('Stuff1.png')
	# diff = cv2.absdiff(img1, background)
	# diff_thresh = cv2.absdiff(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(background,cv2.COLOR_BGR2GRAY))
	# cv2.imshow('Difference', diff); cv2.waitKey(10000); cv2.destroyAllWindows()
	# gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
	# showImage(gray)
	# ret,thresh1 = cv2.threshold(gray,60,255,cv2.THRESH_BINARY)
	# cv2.imwrite('subtracted.png', thresh1)
	# mask = cv2.bitwise_not(thresh1)




	r = removeBackground('Background_only.png')
	# fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	# out_file = cv2.VideoWriter('subtracted_background.avi',fourcc, 20.0, (640,480), False)
	# # out_file = cv2.VideoWriter('huh.avi',fourcc, 20.0, (480, 640), True)
	# ret = True
	# while ret:
	# 	ret, img = camera.read()
	# 	if ret:
	# 		ret, back = r.removeBackground(img)
	# 		out_file.write(back)
	# 		# pdb.set_trace()
	# 		cv2.imshow('Frame', back)
	# 		if cv2.waitKey(1) & 0xFF == ord('q'):
	# 			break
	# out_file.release()

	# img = cv2.imread('Stuff1.png')
	# img_gray = cv2.cvtColor(cv2.imread('subtracted.png'),cv2.COLOR_BGR2GRAY)
	# img_blur = cv2.medianBlur(img_gray, 11)
	# img2, contours, hierarchy = cv2.findContours(img_blur,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# # img2, contours, hierarchy = cv2.findContours(img_blur,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	# indxs = [i for i in range(len(contours)) if len(contours[i]) > 10]
	# # img_contours = cv2.drawContours(img, contours, -1, (0,255,0), 3)

	# # find largest contour
	# object_cnt_idx = np.argmax([cv2.contourArea(cnt) for cnt in contours])
	# img_contours = cv2.drawContours(img, [contours[object_cnt_idx]], 0, (0,255,0), 3)
	# cv2.imshow('Frame', img_contours)
	# cv2.waitKey(10000)
	# for cont,h in zip(contours, hierarchy):
	# 	# pdb.set_trace()
	# 	if all(h[3,:] != -1):
	# 		# this contour has parents!
	# 		continue
	# 	img_contours_indiv = cv2.drawContours(img, [cont], 0, (0,255,0), 3)
	# 	cv2.imshow('Frame', img_contours_indiv)
	# 	if cv2.waitKey(100) & 0xFF == ord('q'):
	# 				break

		
	# oT = objectTracking()
	# ret = True
	# while ret:
	# 	ret, img = camera.read()
	# 	if ret:
	# 		ret, back = r.removeBackground(img)
	#		[x,y], cnt = oT.estimatePosition(back)
	# 		img_box = cOT.drawBoundingBox(img, bbox)
	# 		if len(cnt) != 0:
	# 			img_contours = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
	# 			cv2.imshow('Frame', back)
	# 		# out_file.write(back)
	# 		# pdb.set_trace()
	# 		if cv2.waitKey(1) & 0xFF == ord('q'):
	# 			break
	# # out_file.release()
	# # need an extra processing step to remove the hand from the position estimate
	# img = cv2.imread('0396_frame.png')
	# ret, back = r.removeBackground(img)
	# [x,y], cnt = oT.estimatePosition(back)
	# img_circ = cv2.circle(img, (x,y), 50, (0,255,0), thickness = 5)
	# cv2.imshow('Frame', back)
	# cv2.waitKey(1000)


