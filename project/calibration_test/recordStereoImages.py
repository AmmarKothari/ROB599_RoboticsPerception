import rospy
import rosbag
import sensor_msgs.msg
import pdb
import cv2
import roslaunch
import cv_bridge # CvBridge, CvBridgeError
import subprocess, os

topics = ['/left/image_raw', '/right/image_raw']
# how to force one camera to be the left topic?
# control with how uvc is launched?
# figure out how to turn off auto focus!!!

# launch with rosrun uvc_camera uvc_stereo_node
class recordStereoImages():
	def __init__(self, fn):
		self.bag = rosbag.bag.Bag(fn, mode='w', compression='lz4')

	def startNode(self):
		rospy.init_node('StereoRecorder', anonymous=True)

		# i didn't test these
	def launchStereoNode(self):
		# turns off autofocus -- make sure these are the right cameras
		subprocess.Popen(['v4l2-ctl', '-c', 'focus_auto=0', '-d', '/dev/video0'])
		subprocess.Popen(['v4l2-ctl', '-c', 'focus_auto=0', '-d', '/dev/video1'])
		package = 'uvc_camera'
		executable = 'uvc_stereo_node'
		self.node = roslaunch.core.Node(package, executable)
		self.launch = roslaunch.scriptapi.ROSLaunch()
		self.launch.start()
		self.process = self.launch.launch(self.node)

	def closeStereoNode():
		self.process.stop()

	def record(self):
		data = self.getMessages(topics)
		self.writeToBag(data)

	def getMessages(self, topics):
		d = dict.fromkeys(topics)
		for t in topics:
			d[t] = rospy.wait_for_message(t, sensor_msgs.msg.Image)
		return d

	def writeToBag(self, data):
		for t in topics:
			self.bag.write(t, data[t])

	def shutdown(self):
		self.bag.close()



class extractImagesFromBag():
	def __init__(self, fn):
		self.bagFileNameCheck(fn)
		self.img_count = 0

	def openBag(self):
		self.bag = self.loadBag()
		self.bag_gen = self.bag.read_messages()


	def loadBag(self): # load bag file into object
		# does some error correction
		try:
			b = rosbag.Bag(self.fn, "r")
		except rosbag.bag.ROSBagUnindexedException:
			print "unindexed bag... Trying to reindex."
			p = subprocess.Popen(['rosbag', 'reindex', self.fn])
			try:
				b = rosbag.Bag(self.fn, "r")
			except:
				print "Could not reindex and open "
				raise IOError
		print("Bag File Loaded: %s" %self.fn)
		return b

	def bagFileNameCheck(self, fn): # check file name
		if '.bag' not in fn:
			fn += '.bag' # add .bag if it isn't in the file name?  I suppose you could have a file that doesn't end in bag but that seems confusing
		self.fn = fn

	def iterateThroughBag(self, folderName='.'):
		# assumes sequential order of pairs
		bridge = cv_bridge.CvBridge()
		for topic, msg, t in self.bag_gen:
			cv_image = bridge.imgmsg_to_cv2(msg)
			i = self.img_count/2
			if 'left' in topic:
				img_t = 'left'
			else:
				img_t = 'right'
			fn_name = os.path.join(folderName, 'frame-%04d.%s.png' %(i,img_t))
			self.img_count += 1
			cv2.imwrite(fn_name, cv_image)




if __name__ == '__main__':
	r = recordStereoImages('stereo_smallcheckerboard.bag')
	r.startNode()
	r.launchStereoNode()
	rate = rospy.Rate(10)


	while not rospy.is_shutdown():
		try:
			r.record()
			rate.sleep()
		except rospy.ROSInterruptException:
			r.shutdown()
	rospy.on_shutdown(r.shutdown)

	print('Shutting Down!')