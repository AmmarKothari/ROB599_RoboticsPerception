import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb
from utils import horizConcat, avgImage


file1 = '/test_pics/fig1.png'
file2 = '/test_pics/fig4.png'
file3 = '/test_pics/fig21.png'

#todo - k means segmentation
#1 - figure out k means clustering output
#2 - segment out the object
#3 - detect gap left by object from image if we crop object


#thought - subtract object area from the background. In the object spot, reassign it with a new label and we are done!

class imgSegment(object):
	def __init__(self):
		i = 1
		self.change_flag = True

	def blob(self, image):
		# Setup SimpleBlobDetector parameters.
		params = cv2.SimpleBlobDetector_Params()
		 
		# Change thresholds
		params.minThreshold = 10;
		params.maxThreshold = 255;

		#filter by color
		params.filterByColor = False
		 
		# Filter by Area.
		params.filterByArea = True
		params.minArea = 10000
		params.maxArea = 10000000000 #got values empirically
		 
		# Filter by Circularity
		params.filterByCircularity = False
		params.minCircularity = 0.1
		 
		# Filter by Convexity
		params.filterByConvexity = False
		params.minConvexity = 0.8
		 
		# Filter by Inertia
		params.filterByInertia = False
		params.minInertiaRatio = 0.1

		#actually set up the blob detector
		detector = cv2.SimpleBlobDetector_create(params)
		#detector_all = cv2.simpleBlobDetector_create()
		 
		# Detect blobs.
		keypoints = detector.detect(image)
		#keypoints_all = detector_all.detect(small)
		print(keypoints[0].pt) 
		 
		# Draw detected blobs as red circles.
		# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
		small_extra = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		return small_extra

	def blob_user(self, image_list):
		window = cv2.namedWindow("blob", cv2.WINDOW_NORMAL)
		frame = cv2.imread(image_list[0])
		image_counter = 1;
		# Setup SimpleBlobDetector parameters.
		params = cv2.SimpleBlobDetector_Params()
		 
		# Change thresholds
		params.minThreshold = 10;
		params.maxThreshold = 255;

		#filter by color
		params.filterByColor = False
		 
		# Filter by Area.
		params.filterByArea = True
		params.minArea = 10000
		params.maxArea = 10000000000 #got values empirically
		 
		# Filter by Circularity
		params.filterByCircularity = False
		params.minCircularity = 0.1
		 
		# Filter by Convexity
		params.filterByConvexity = False
		params.minConvexity = 0.8
		 
		# Filter by Inertia
		params.filterByInertia = False
		params.minInertiaRatio = 0.1

		
		flag = True

		# Setup all the trackbars!
		cv2.createTrackbar('minThreshold','blob',0,255,self.nothing)
		cv2.createTrackbar('maxThreshold','blob',0,255,self.nothing)
		cv2.createTrackbar('filterByColor','blob',0,1,self.nothing)
		cv2.createTrackbar('filterByArea','blob',0,1,self.nothing)
		cv2.createTrackbar('minArea','blob',100,1000000,self.nothing)
		cv2.createTrackbar('maxArea','blob',100000,100000000,self.nothing)
		cv2.createTrackbar('filterByCircularity','blob',0,1,self.nothing)
		cv2.createTrackbar('minCircularity','blob',0,100,self.nothing)

		# set default params
		cv2.setTrackbarPos('minThreshold', 'blob', 10)
		cv2.setTrackbarPos('maxThreshold', 'blob', 255)
		cv2.setTrackbarPos('filterByColor', 'blob', 0)
		cv2.setTrackbarPos('filterByArea', 'blob', 1)
		cv2.setTrackbarPos('minArea', 'blob', 10000)
		cv2.setTrackbarPos('maxArea', 'blob', 100000000)
		cv2.setTrackbarPos('filterByCircularity', 'blob', 0)
		cv2.setTrackbarPos('minCircularity', 'blob', 10)

		while flag:
			params.minThreshold = cv2.getTrackbarPos('minThreshold','blob');
			params.maxThreshold = cv2.getTrackbarPos('maxThreshold','blob');
			params.filterByColor = cv2.getTrackbarPos('filterByColor','blob');
			params.filterByArea = cv2.getTrackbarPos('filterByArea','blob');
			params.minArea = cv2.getTrackbarPos('minArea','blob');
			params.maxArea = cv2.getTrackbarPos('maxArea','blob');
			params.filterByCircularity = cv2.getTrackbarPos('filterByCircularity','blob');
			params.minCircularity = cv2.getTrackbarPos('minCircularity','blob')/100.0;

			#actually set up the blob detector
			detector = cv2.SimpleBlobDetector_create(params)
			#detector_all = cv2.simpleBlobDetector_create()
			 
			# Detect blobs.
			keypoints = detector.detect(frame)
			#keypoints_all = detector_all.detect(small)
			# print(keypoints[0].pt) 
			 
			# Draw detected blobs as red circles.
			# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
			frame_blob = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			cv2.imshow("blob", frame_blob)
			print("Ready")
			while True:
				if cv2.waitKey(10) & 0xFF == ord('q'):
					# break out of loop
					flag = False
					break
				elif cv2.waitKey(10) & 0xFF == ord('c'):
					# continue to next image
					frame = cv2.imread(image_list[image_counter])
					image_counter += 1
					print('Going to next image')
					break
				elif cv2.waitKey(10) & 0xFF == ord('r'):
					# redo the calculation
					print('Redoing Calculation')
					break



		return frame_blob

	def nothing(self, x):
		self.change_flag = True

	def colorSegmentFilter_user(self, image_list):
		window = cv2.namedWindow("colorSegment", cv2.WINDOW_NORMAL)
		frame = cv2.imread(image_list[0])
		image_counter = 1;

		cv2.createTrackbar('H','colorSegment',0,255,self.nothing)
		cv2.createTrackbar('S','colorSegment',0,255,self.nothing)
		cv2.createTrackbar('V','colorSegment',0,255,self.nothing)
		cv2.createTrackbar('R','colorSegment',0,255,self.nothing)
		cv2.createTrackbar('G','colorSegment',0,255,self.nothing)
		cv2.createTrackbar('B','colorSegment',0,255,self.nothing)


		cv2.setTrackbarPos('H', 'colorSegment', 31)
		cv2.setTrackbarPos('S', 'colorSegment', 91)
		cv2.setTrackbarPos('V', 'colorSegment', 69)
		cv2.setTrackbarPos('R', 'colorSegment', 7)
		cv2.setTrackbarPos('G', 'colorSegment', 71)
		cv2.setTrackbarPos('B', 'colorSegment', 50)


		while True:
			# get values
			H = cv2.getTrackbarPos('H','colorSegment');
			S = cv2.getTrackbarPos('S','colorSegment');
			V = cv2.getTrackbarPos('V','colorSegment');
			R = cv2.getTrackbarPos('R','colorSegment');
			G = cv2.getTrackbarPos('G','colorSegment');
			B = cv2.getTrackbarPos('B','colorSegment');

			if self.change_flag:
				self.change_flag = False
				# turn into HSV image
				R_img = frame[:,:,0]
				G_img = frame[:,:,1]
				B_img = frame[:,:,2]
				HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # 3 channels
				H_img = HSV[:,:,0]
				S_img = HSV[:,:,1]
				V_img = HSV[:,:,2]

				ret,H_thresh = cv2.threshold(H_img,H,255,cv2.THRESH_BINARY)
				ret,S_thresh = cv2.threshold(S_img,S,255,cv2.THRESH_BINARY)
				ret,V_thresh = cv2.threshold(V_img,V,255,cv2.THRESH_BINARY)
				ret,R_thresh = cv2.threshold(R_img,R,255,cv2.THRESH_BINARY)
				ret,G_thresh = cv2.threshold(G_img,G,255,cv2.THRESH_BINARY)
				ret,B_thresh = cv2.threshold(B_img,B,255,cv2.THRESH_BINARY)

				HSV_masked = HSV
				RGB_masked = frame
				for t in (H_thresh, S_thresh, V_thresh, R_thresh, G_thresh, B_thresh):

					HSV_masked = cv2.bitwise_and(HSV_masked,HSV_masked,mask = t)
					RGB_masked = cv2.bitwise_and(RGB_masked,RGB_masked,mask = t)

				HSV_thresh = np.stack((H_thresh, S_thresh, V_thresh), axis=2)
			# pdb.set_trace()
			# Draw detected blobs as red circles.
			# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
			# frame_blob = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			# cv2.imshow("colorSegment", HSV)
				cv2.imshow("colorSegment", horizConcat((HSV_thresh, H_thresh, S_thresh, V_thresh, R_thresh, G_thresh, B_thresh, HSV_masked, RGB_masked)))
			# print("Ready")
			if cv2.waitKey(50) & 0xFF == ord('q'):
				# break out of loop
				break
			elif cv2.waitKey(50)  & 0xFF == ord('s'):
				print("Image Saved")
				cv2.imwrite("{}_HSV_colorseg.jpg".format(image_counter), HSV_masked)
				cv2.imwrite("{}_RGB_colorseg.jpg".format(image_counter), RGB_masked)

			elif cv2.waitKey(50) & 0xFF == ord('c'):
			# 	# continue to next image
				frame = cv2.imread(image_list[image_counter])
				image_counter += 1
				print('Going to next image')
				self.change_flag = True
				# 	break
				# elif cv2.waitKey(10) & 0xFF == ord('r'):
				# 	# redo the calculation
				# 	print('Redoing Calculation')
				# 	break

		print("H: {}, S: {}, V: {}, R: {}, G: {}, B: {}".format(H, S, V, R, G, B))


	# def watershed_user(self, image_list):
	# 	window = cv2.namedWindow("watershed", cv2.WINDOW_NORMAL)
	# 	frame = cv2.imread(image_list[0])
	# 	image_counter = 1;












if __name__ == '__main__': 
	im = cv2.imread('fig6.jpg')
	img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


	# =============================================================
	# add image blur to help the k means clustering?
	#
	i = cv2.bilateralFilter(img,9,150,150)
	#i = cv2.GaussianBlur(img,(7,7),0)
	#i = cv2.blur(img, (10,10))

	Z = i.reshape((-1,3))
	#Z = cv2.pyrMeanShiftFiltering( Z, res, 20, 45, 3); 	
	#Z = cv2.GaussianBlur(Z, Z, Size(5,5), 2, 2);

	# convert to np.float32
	Z = np.float32(Z)


	# =============================================================
	# k means clustering on the prepared image?
	#
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 2
	ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	 
	small = cv2.resize(res2, (0,0), fx=0.75, fy=0.75) 
	#small_extra = cv2.threshold(small,220,255,0)


	# =============================================================
	# Set up the blob detector with default parameters.
	#

	

	cv2.imshow('image after segmentation',small_extra) 

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# =============================================================
	# Try to find contours in the image
	#
#	contours, hierarchy, hmmmm = cv2.findContours(small,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#	ctr = np.array(contours).reshape((-1,1,2)).astype(np.int32)
#
#	cv2.drawContours(small,[ctr],0,(0,100,100),-1)
#
#	cv2.imshow('image after segmentation',small) 
#
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()

	# =============================================================
	# try canny edge detection on the blobs to try to separate them?
	#
	edges = cv2.Canny(small,100,200)

	plt.subplot(121),plt.imshow(small,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])

	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.scatter(keypoints[0].pt[0],keypoints[0].pt[1])
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

	plt.show()