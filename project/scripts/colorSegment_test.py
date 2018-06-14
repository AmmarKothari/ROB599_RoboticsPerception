from segment import imgSegment
import cv2
import glob
import os
import pdb

from utils import horizConcat, avgImage

seg = imgSegment()


imageFolder_path = '../Data/Pictures_of_ball/20180419_153*.jpg'
images = glob.glob(imageFolder_path)

seg.colorSegmentFilter_user(images)

# cap = cv2.VideoCapture(os.path.abspath(imageFolder_path))

# ret = True;
# while(ret):
# window = cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
# for fn in images:
# 	frame = cv2.imread(fn)
# 	blob = seg.blob(frame)

# 	cv2.imshow('frame', horizConcat(frame, blob))
# 	if cv2.waitKey(10) & 0xFF == ord('q'):
# 		break



