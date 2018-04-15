# quick script to capture images with a webcam


# start webcam with rosrun uvc_camera uvc_camera_node 

import pdb, os
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class recordImages(object):
	def __init__(self, fn):
		self.node = rospy.init_node('camera_capture', anonymous=True)
		self.rate = rospy.Rate(10) #10 Hz
		self.bridge = CvBridge()
		self.fn = ''
		self.i = 0


	def userInput(self):
		# waits for user input to collect and save an image
		loop_flag = True
		while loop_flag:
			collect_flag = raw_input('Hit Enter to collect Data. Hit Q and then enter to collect data.')
			if collect_flag == '': # user hit enter
				self.saveImage()
			elif collect_flag.lower() == 'q':
				loop_flag = False


	def saveImage(self):
		msg = rospy.wait_for_message('/image_raw', Image)
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
		except CvBridgeError as e:
			print(e)

		# cv2.imshow('Image Window', cv_image)
		# cv2.waitKey(3)
		cv2.imwrite(os.path.join(self.fn, 'Image%s.png' %self.i), cv_image)
		self.i += 1



if __name__ == '__main__':
	# pdb.set_trace()
	R = recordImages('test')
	R.userInput()
	# while not rospy.is_shutdown():
	# 	R.saveImage()
	# 	R.rate.sleep()