import cv2
import pdb


def drawBB(bbox, color):
	p1 = (int(bbox[0]), int(bbox[1]))
	p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
	cv2.rectangle(frame, p1, p2, color, 2, 1)



video = cv2.VideoCapture("CokeCanTrackingShort.avi")
ret, frame = video.read()

out_fn = "CokeCanTrackingShort_WithTracking.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')


out_file = cv2.VideoWriter(out_fn,fourcc, 20.0, (frame.shape[1],frame.shape[0]), True)


tracker1 = cv2.TrackerKCF_create()
tracker2 = cv2.TrackerTLD_create()
bbox = cv2.selectROI(frame, False)

color1 = (255,0,0)
color2 = (0,0,255)

# Initialize tracker with first frame and bounding box
ok1 = tracker1.init(frame, bbox)
ok2 = tracker2.init(frame, bbox)



while True:
	# Read a new frame
	ok, frame = video.read()
	if not ok:
		break


	# Update tracker
	ok1, bbox1 = tracker1.update(frame)
	ok2, bbox2 = tracker2.update(frame)


	# Draw bounding box
	if ok1: # both succeed 
		if ok2:
			drawBB(bbox2, color2)
			drawBB(bbox1, color1)
	else:
		# Tracking failure
		cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)


	# Display result
	cv2.imshow("Tracking", frame)
	cv2.waitKey(10)
	out_file.write(frame)

out_file.release()
cv2.destroyAllWindows()


