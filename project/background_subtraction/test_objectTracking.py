import cv2




video = cv2.VideoCapture('../calibration_test/object_tracking.avi')
tracker = cv2.TrackerKCF_create()

# ret, frame = video.read()
f0 = cv2.imread('../Objet_tracking/0001_frame.png')
f1 = cv2.imread('../Objet_tracking/0192_frame.png')
f2 = cv2.imread('../Objet_tracking/0316_frame.png')

# bbox = cv2.selectROI(frame, False)
bboxs = [cv2.selectROI(frame, False) for frame in [f0, f1, f2]]

ok = tracker.init(frame, bbox)

ret = True
while ret:
	ret, frame = video.read()
	if ret:
		ok, bbox = tracker.update(frame)
	if ok:
		# Tracking success
		p1 = (int(bbox[0]), int(bbox[1]))
		p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
		cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
	else :
		# Tracking failure
		cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

	# Display tracker type on frame
	# cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
 
	# Display FPS on frame
	# cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

	# Display result
	cv2.imshow("Tracking", frame)

	# Exit if ESC pressed
	k = cv2.waitKey(1) & 0xff
	if k == 27 : break