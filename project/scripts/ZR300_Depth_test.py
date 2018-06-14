

import cv2, os, pdb, glob
import numpy as np
## setup logging
import logging
logging.basicConfig(level = logging.INFO)

## import the package
import pyrealsense as pyrs
from pyrealsense import offline

# # # Record Video Test
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_file = cv2.VideoWriter('DepthTESTVIDEO.avi',fourcc, 20.0, (640*2,480), True)
font = cv2.FONT_HERSHEY_SIMPLEX


SAVEFOLDER = 'DepthTest_ZR300'
FPS = 30


color_imgs = glob.glob(os.path.join(SAVEFOLDER, 'color_*.jpg'))
depth_imgs = glob.glob(os.path.join(SAVEFOLDER, 'depth_*.jpg'))

color_imgs.sort(key= lambda x: int(x.split('/')[1].replace('color_','').replace('.jpg','')))
depth_imgs.sort(key= lambda x: int(x.split('/')[1].replace('depth_','').replace('.jpg','')))

frame = cv2.imread(color_imgs[10])

tracker2 = cv2.TrackerKCF_create()
# bbox2 = cv2.selectROI(frame, False)
bbox2 = (361, 250, 63, 158)


offline.load_depth_intrinsics('3481802454')

fn_depth_np = os.path.join(SAVEFOLDER, 'depth_{}.npy'.format(10))
depth_mat = np.load(fn_depth_np)
relevant_depth = depth_mat[bbox2[0]:bbox2[0]+bbox2[2], bbox2[1]:bbox2[1]+bbox2[3]]
avg_depth = np.mean(relevant_depth)*offline.depth_scale ## depth in centimeters
cv2.rectangle(frame,(bbox2[0],bbox2[1]),(bbox2[0]+bbox2[2],bbox2[1]+bbox2[3]),(0,255,0),3)
cv2.putText(frame, "Average Depth in Box: {0:.3f} m".format(avg_depth), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
depth_img = cv2.imread(depth_imgs[10])

depth_img[0:bbox2[1],:,:] = 0
depth_img[:,0:bbox2[0],:] = 0
depth_img[bbox2[0]+bbox2[2]:,:,:] = 0
depth_img[:,bbox2[1]+bbox2[3]:,:] = 0

frame[0:bbox2[1],:,:] = 0
frame[:,0:bbox2[0],:] = 0
frame[bbox2[0]+bbox2[2]:,:,:] = 0
frame[:,bbox2[1]+bbox2[3]:,:] = 0

cv2.imshow('', frame); cv2.waitKey(0)
cv2.imshow('', depth_img); cv2.waitKey(0)

cv2.imshow('', frame)
cv2.imwrite('ExampleDepthImage_Color.jpg', frame)
cv2.imwrite('ExampleDepthImage_Depth.jpg', depth_img)
cv2.waitKey(1000)




# for i in range(10,len(color_imgs)):
# 	frame = cv2.imread(color_imgs[i])
# 	ok2, bbox2 = tracker2.update(frame)
# 	pdb.set_trace()
# 	if ok2:
# 		fn_depth_np = os.path.join(SAVEFOLDER, 'depth_{}.npy'.format(i))
# 		depth_mat = np.load(fn_depth_np)
# 		pdb.set_trace()
# 	else:
# 		# Tracking failure
# 		cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

# 	cv2.imshow('',frame)
# 	if cv2.waitKey(100) == ord('q'):
# 		break
