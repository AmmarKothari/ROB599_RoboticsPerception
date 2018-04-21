import rospy
import rosbag
import sensor_msgs.msg
import pdb
import cv2
import os

import cv_bridge # CvBridge, CvBridgeError

topics = ['/camera/rgb/image_raw', '/camera/depth/image_rect_raw']

class recordDepthImages():
	def __init__(self, fn):
		self.bag = rosbag.bag.Bag(fn, mode='w', compression='lz4')

	def startNode(self):
		rospy.init_node('DepthRecorder', anonymous=True)

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
			if 'rgb' in topic:
				img_t = 'color'
			else:
				img_t = 'depth'
			fn_name = os.path.join(folderName, 'frame-%04d.%s.png' %(i,img_t))
			self.img_count += 1
			cv2.imwrite(fn_name, cv_image)










if __name__ == '__main__':
	r = recordDepthImages('Depth_Object.bag')
	r.startNode()
	rate = rospy.Rate(10)


	while not rospy.is_shutdown():
		r.record()
		try:
			rate.sleep()
		except rospy.ROSInterruptException:
			r.shutdown()
	rospy.on_shutdown(r.shutdown)

	print('Shutting Down!')
	pdb.set_trace()
