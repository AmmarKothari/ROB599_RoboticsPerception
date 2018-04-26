'''

Test images from depth map

'''



from DepthFromStero import DepthFromStereo
import glob
import os
import cv2

folder = "/home/ammar/Documents/Projects/ROB599_RoboticsPerception/project/calibration_test/StereoImages/WithoutObject/Pairs"

left_fns = glob.glob(os.path.join(folder, '*.left.png'))
right_fns = glob.glob(os.path.join(folder, '*.right.png'))

left_img = cv2.imread(left_fns[0]) 
right_img = cv2.imread(right_fns[0]) 

D = DepthFromStereo()
disparity_img = D.combineImagesIntoDisparity(left_img, right_img)
D.showImg(disparity_img)