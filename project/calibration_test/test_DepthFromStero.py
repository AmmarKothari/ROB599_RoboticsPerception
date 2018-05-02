'''

Test images from depth map

'''



from DepthFromStero import DepthFromStereo, phonyStereoCamera, calibrateStereoCameras, horizConcat, stereoCamera
import glob
import os
import cv2

folder = "/home/ammar/Documents/Projects/ROB599_RoboticsPerception/project/calibration_test/StereoImages/WithoutObject/Pairs"

SC = phonyStereoCamera(folder, folder)
SC = stereoCamera()

cal = calibrateStereoCameras()
if not os.path.isfile('checkerboardFeatures.npz'):
	cal.extractCheckerboardFeatures(SC)

cal.loadCheckerboardFeatures('checkerboardFeatures.npz')
left, right = SC._getCurrentStereoPair()
if not os.path.isfile('CameraCalRight.npz') and not os.path.isfile('CameraCalLeft.npz'):
	cal.calibrateStereoCameras(cal.grayscaleImage(left))

cal.loadCameraParameters('CameraCalLeft.npz', 'CameraCalRight.npz')
if not os.path.isfile('StereoCal.npz'):
	cal.extrinsicCalibrateStereoCameras(cal.grayscaleImage(left))
cal.loadExtrinsicParameters('StereoCal.npz')
if not os.path.isfile('RectifyCal.npz'):
	cal.rectifyCalibrate(cal.grayscaleImage(left))
cal.loadRectifyParams('RectifyCal.npz')
loop_flag = True

while loop_flag:
	left, right, loop_flag = SC.getStereoPair()
	left_rect, right_rect = cal.rectifyImages(left, right)
	cv2.imshow('test', horizConcat(left_rect, right_rect))
	cv2.waitKey(40)
cv2.destroyAllWindows()



# left_fns = glob.glob(os.path.join(folder, '*.left.png'))
# right_fns = glob.glob(os.path.join(folder, '*.right.png'))

# left_img = cv2.imread(left_fns[0]) 
# right_img = cv2.imread(right_fns[0]) 

# D = DepthFromStereo()
# disparity_img = D.combineImagesIntoDisparity(left_img, right_img)
# D.showImg(disparity_img)