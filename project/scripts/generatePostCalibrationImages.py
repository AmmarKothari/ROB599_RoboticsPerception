### Use Camera Calibration to generate rectified images


import cv2
from cameraCal import calibrateCameraCustom, undistortImage
from utils import horizConcat, avgImage
import glob
import os
import pdb

# pdb.set_trace()
imageFolder_path = '../Data/StereoImages/Calibration/RGBPairs'
imageFolder_path = os.path.join(imageFolder_path, '%4d_left.png')
images = glob.glob(imageFolder_path)

cap = cv2.VideoCapture(imageFolder_path)
cal = calibrateCameraCustom(cap, w=7, h=9);

ret = True
RECAL_COUNT = 10
if not os.path.isfile('CameraCal.npz'):

	while(ret):
		ret, frame = cap.read()
		if ret:
			cal.addImageToCal(frame)
			# undist = undistortImage(cal)
			# undistFrame = undist.undistortRectify(frame)
			# pdb.set_trace()
			# if not any([i == 0 for i in undistFrame.shape]): # skip if 0 on one chanell
			if True:
				# cv2.imshow('frame', horizConcat(frame, undistFrame))
				cv2.imshow('frame', frame)
				if cv2.waitKey(10) & 0xFF == ord('q'):
					break
		else:
			print('Camera Failed to open')
	cal.cal()
	cap = cv2.VideoCapture(imageFolder_path)
else:
	cal.loadCameraParams('CameraCal.npz')
undist = undistortImage(cal)

ret, frame = cap.read()
undistFrame, undist_success = undist.undistortRectify(frame)

if undist_success:
	# cv2.imshow('frame', horizConcat((frame, undistFrame)))
	cv2.imshow('frame', avgImage(frame, undistFrame))
	cv2.waitKey(100)
	font = cv2.FONT_HERSHEY_SIMPLEX
	# pdb.set_trace()
	cv2.putText(frame,'Before Calibration: {} x {}'.format(*frame.shape[0:2]),(10,500), font, 2,(0,255,0),2,cv2.LINE_AA)
	cv2.imwrite('PreCal.png', frame)
	cv2.putText(undistFrame,'After Calibration: {} x {}'.format(*undistFrame.shape[0:2]),(10,500), font, 2,(0,255,0),2,cv2.LINE_AA)
	cv2.imwrite('PostCal.png', undistFrame)


# ret = True
# while(ret):
# 	ret, frame = cap.read()
# 	if ret:
# 		undistFrame = undist.undistortRectify(frame)
# 		# pdb.set_trace()
# 		if not any([i == 0 for i in undistFrame.shape]): # skip if 0 on one chanell
# 		# if True:
# 			cv2.imshow('frame', horizConcat(frame, undistFrame))
# 			# cv2.imshow('frame', frame)
# 			if cv2.waitKey(100) & 0xFF == ord('q'):
# 				break
# 	else:
# 		print('Camera Failed to open')

cap.release()
cv2.destroyAllWindows()