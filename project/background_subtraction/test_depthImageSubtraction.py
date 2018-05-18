'''
test out removing background for a depth image


'''

import cv2
from backgroundSubtraction import removeBackground, cvObjectTracking, objectDetection, showImage, convert_z16_to_bgr

background_depth_file = '../Data/RGBDTestScene/Attempt3/   9_depth.png'
object_depth_file = '../Data/RGBDTestScene/Attempt3/ 250_depth.png'

img = cv2.imread(object_depth_file)


r = removeBackground(background_depth_file)
r.background = convert_z16_to_bgr(cv2.cvtColor(r.background,cv2.COLOR_BGR2GRAY))
ret, back = r.maskBackground(img)

back_1 = cv2.cvtColor(back,cv2.COLOR_BGR2GRAY)
color_depth = convert_z16_to_bgr(back_1)
showImage(color_depth)
