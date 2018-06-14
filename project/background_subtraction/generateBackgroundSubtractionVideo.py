import cv2
from backgroundSubtraction import removeBackground, cvObjectTracking, objectDetection, showImage, convert_z16_to_bgr
import sys
sys.path.insert(0, '../scripts')
from cameraCal import calibrateCameraCustom
from utils import horizConcat
import pdb



r = removeBackground('Images/Background_only.png')
fgbg = cv2.createBackgroundSubtractorMOG2()


camera = cv2.VideoCapture('Images/background_subtraction.avi')



fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_file = cv2.VideoWriter('BackgroundRemovalPresentation.avi', fourcc, 30, (r.background.shape[1]*3, r.background.shape[0]))


# recording video of background being removed from scene
retc = True
i = 0
while retc:
	retc, img = camera.read()
	i += 1
	if retc:
		ret, back = r.maskBackground(img)
		fgmask = fgbg.apply(img)
		if ret:
			# pdb.set_trace()
			cv2.imshow('Frame', horizConcat((img, back, fgmask)))
			out_file.write(horizConcat((img, back, fgmask)))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		print('Failed to grab frame {}'.format(i))

out_file.release()
cv2.destroyAllWindows()