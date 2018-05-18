import cv2
from backgroundSubtraction import removeBackground, cvObjectTracking, objectDetection, showImage, convert_z16_to_bgr





r = removeBackground('Background_only.png')
camera = cv2.VideoCapture('background_subtraction.avi')
font = cv2.FONT_HERSHEY_SIMPLEX




### testing for object tracking with built in openCV
ret, back = r.maskBackground(cv2.imread('Stuff1.png'))
cOT = cvObjectTracking((149, 85, 189, 227))
oD = objectDetection()
retc = True
i = 0
while retc:
	retc, img = camera.read()
	# pdb.set_trace()
	i += 1
	if retc:
		ret, back = r.maskBackground(img)
		[x,y], cnt = oD.estimatePosition(back)
		print("Contours {}, {}".format(x, y))
		if len(cnt) != 0:
			oD.getBoundingRect
			cOT = cvObjectTracking((149, 85, 189, 227))
			ret, bbox = cOT.trackObject(back)

			if ret:
				img_box = cOT.drawBoundingBox(img, bbox)
				pdb.set_trace()
				text = 'Frame {}, BBOX: ({}, {}, {}, {})' %(i, bbox) 
				img_text = cv2.putText(img_box,text,(10,400), font, 2,(255,255,255),2,cv2.LINE_AA)
				cv2.imshow('Frame', img_text)
			else:
				img_contours = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
				cv2.imshow('Frame', img_contours)

		else:
			# pdb.set_trace()
			img_text = cv2.putText(img,'Frame %s' %i,(10,400), font, 2,(255,255,255),2,cv2.LINE_AA)
			cv2.imshow('Frame', img_text)
		# out_file.write(back)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		print('Failed to grab frame {}'.format(i))