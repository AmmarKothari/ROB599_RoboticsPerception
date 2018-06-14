'''

Test images from depth map

'''



from DepthFromStero import DepthFromStereo, phonyStereoCamera, calibrateStereoCameras, horizConcat, stereoCamera
import glob
import os
import cv2
import pdb

# folder = "/home/ammar/Documents/Projects/ROB599_RoboticsPerception/project/calibration_test/StereoImages/WithoutObject/Pairs"
# folder = "/home/ammar/Documents/Projects/ROB599_RoboticsPerception/project/calibration_test/StereoImages/Calibration"
folder = "/home/ammar/Documents/Projects/ROB599_RoboticsPerception/project/Data/StereoImages/Calibration2"

SC = phonyStereoCamera(os.path.join(folder, 'RGBPairs'), os.path.join(folder, 'RGBPairs'))
# SC = stereoCamera()
SC.crop_flag = False
cal = calibrateStereoCameras()
cal.w = SC.w
cal.h = SC.h
cal.img_w = cal.w
cal.img_h = cal.h
if not os.path.isfile('checkerboardFeatures.npz'):
	cal.extractCheckerboardFeatures(SC)

SC.crop_flag = True
cal.loadCheckerboardFeatures('checkerboardFeatures.npz')

left, right, loop_flag = SC._getCurrentStereoPair()
if not os.path.isfile('CameraCalRight.npz') and not os.path.isfile('CameraCalLeft.npz'):
	cal.calibrateStereoCameras(cal.grayscaleImage(left))	

cal.loadCameraParameters('CameraCalLeft.npz', 'CameraCalRight.npz')
if not os.path.isfile('StereoCal.npz'):
	cal.extrinsicCalibrateStereoCameras(cal.grayscaleImage(left))
cal.loadExtrinsicParameters('StereoCal.npz')

# if not os.path.isfile('RectifyCal.npz'):
if True:
	cal.rectifyCalibrate(cal.grayscaleImage(left))
	i = 0
	while loop_flag:
		left, right, loop_flag = SC.getStereoPair()
		rectL, rectR = cal.rectifyImages(left, right)
		cv2.imwrite(os.path.join(folder, 'Rectified', '%04d_left.png' %i), rectL)
		cv2.imwrite(os.path.join(folder, 'Rectified', '%04d_right.png' %i), rectR)

		i += 1


cal.loadRectifyParams('RectifyCal.npz')
loop_flag = True

# while loop_flag:
# 	left, right, loop_flag = SC.getStereoPair()
# 	left_rect, right_rect = cal.rectifyImages(left, right)
# 	cv2.imshow('test', horizConcat(left_rect, right_rect))
# 	cv2.waitKey(40)
# cv2.imwrite('Rectified.png', horizConcat(left_rect, right_rect))
# cv2.destroyAllWindows()
i = 0
while loop_flag:
	left, right, loop_flag = SC.getStereoPair()
	disp = cal.disparityImage(left, right)
	# pdb.set_trace()
	cv2.imshow('test', horizConcat((disp, left, right)))
	cv2.imwrite(os.path.join(folder, 'Disparity', '%04d.png' %i), disp)
	cv2.waitKey(40)
	i +=1
cv2.imwrite('Disparity.png', horizConcat((disp, left, right)))
cv2.destroyAllWindows()


# left_fns = glob.glob(os.path.join(folder, '*.left.png'))
# right_fns = glob.glob(os.path.join(folder, '*.right.png'))

# left_img = cv2.imread(left_fns[0]) 
# right_img = cv2.imread(right_fns[0]) 

# D = DepthFromStereo()
# disparity_img = D.combineImagesIntoDisparity(left_img, right_img)
# D.showImg(disparity_img)